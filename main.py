import os
import logging
from re import template
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

import config
import data_loader
from model import Casrel
from utils.common_utils import set_seed, set_logger, read_json, fine_grade_tokenize
from utils.train_utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model
from utils.metric_utils import calculate_metric_relation, get_p_r_f
from tensorboardX import SummaryWriter

args = config.Args().get_parser()
set_seed(args.seed)
logger = logging.getLogger(__name__)

if args.use_tensorboard == "True":
    writer = SummaryWriter(log_dir='./tensorboard')
  
def get_spo(object_preds, subject_ids, length, example, id2tag):
  # object_preds:[batchsize, maxlen, num_labels, 2]
  num_label = object_preds.shape[2]
  num_subject = len(subject_ids)
  relations = []
  subjects = []
  objects = []
  # print(object_preds.shape, subject, length, example)
  for b in range(num_subject):
    tmp = object_preds[b, ...]
    subject_start, subject_end = subject_ids[b].cpu().numpy()
    subject = example[subject_start:subject_end+1]
    if subject not in subjects:
      subjects.append(subject)
    for label_id in range(num_label):
      start = tmp[:, label_id, :1]
      end = tmp[:, label_id, 1:]
      start = start.squeeze()[:length]
      end = end.squeeze()[:length]
      for i, st in enumerate(start):
        if st > 0.5:
          s = i
          for j in range(i, length):
            if end[j] > 0.5:
              e = j
              object = example[s:e+1]
              if object not in objects:
                objects.append(object)
              if (subject, id2tag[label_id], object) not in relations:
                relations.append((subject, id2tag[label_id], object))
              break
  # print(relations) 
  return relations, subjects, objects
  


def get_subject_ids(subject_preds, mask):
  lengths = torch.sum(mask, -1)
  starts = subject_preds[:, :, :1]
  ends = subject_preds[:, :, 1:]
  subject_ids = []
  for start, end, l in zip(starts, ends, lengths):
    tmp = []
    start = start.squeeze()[:l]
    end = end.squeeze()[:l]
    for i, st in enumerate(start):
      if st > 0.5:
        s = i
        for j in range(i, l):
          if end[j] > 0.5:
            e = j
            if (s,e) not in subject_ids:
              tmp.append([s,e])
            break

    subject_ids.append(tmp)
  return subject_ids

class BertForRe:
    def __init__(self, args, train_loader, dev_loader, test_loader, id2tag, tag2id, model, device):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.id2tag = id2tag
        self.tag2id = tag2id
        self.model = model
        self.device = device
        if train_loader is not None:
          self.t_total = len(self.train_loader) * args.train_epochs
          self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = 500 #每多少个step打印损失及进行验证
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data[:-1]:
                    batch = batch.to(self.device)
                # batch_token_ids, attention_mask, token_type_ids, batch_subject_labels, batch_object_labels, batch_subject_ids
                loss = self.model(batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5])

                # loss.backward(loss.clone().detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))
                global_step += 1
                if self.args.use_tensorboard == "True":
                    writer.add_scalar('data/loss', loss.item(), global_step)
                if global_step % eval_steps == 0:
                    precision, recall, f1_score = self.dev()
                    logger.info('[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(precision, recall, f1_score))
                    if f1_score > best_f1:
                        save_model(self.args, self.model, model_name, global_step)
                        best_f1 = f1_score


    def dev(self):
      self.model.eval()
      spos = []
      true_spos = []
      subjects = []
      objects = []
      all_examples = []
      with torch.no_grad():
          for eval_step, dev_batch_data in enumerate(self.dev_loader):
              for dev_batch in dev_batch_data[:-1]:
                  dev_batch = dev_batch.to(self.device)
              
              seq_output, subject_preds = self.model.predict_subject(dev_batch_data[0], dev_batch_data[1],dev_batch_data[2])
              # 注意这里需要先获取subject，然后再来获取object和关系，和训练直接使用subject_ids不一样
              cur_batch_size = dev_batch_data[0].shape[0]
              dev_examples = dev_batch_data[-1]
              true_spos += [i[1] for i in dev_examples]
              all_examples += [i[0] for i in dev_examples]
              subject_labels = dev_batch_data[3].cpu().numpy()
              object_labels = dev_batch_data[4].cpu().numpy()
              subject_ids = get_subject_ids(subject_preds, dev_batch_data[1])

              example_lengths = torch.sum(dev_batch_data[1].cpu(), -1)
              
              for i in range(cur_batch_size):
                seq_output_tmp = seq_output[i, ...]
                subject_ids_tmp = subject_ids[i]
                length = example_lengths[i]
                example = dev_examples[i][0]
                if subject_ids_tmp:
                  seq_output_tmp = seq_output_tmp.unsqueeze(0).repeat(len(subject_ids_tmp), 1, 1)
                  subject_ids_tmp = torch.tensor(subject_ids_tmp, dtype=torch.long, device=device)
                  if len(seq_output_tmp.shape) == 2:
                    seq_output_tmp = seq_output_tmp.unsqueeze(0)
                  object_preds = model.predict_object([seq_output_tmp, subject_ids_tmp])
                  spo, subject, object = get_spo(object_preds, subject_ids_tmp, length, example, self.id2tag)
                  spos.append(spo)
                  subjects.append(subject)
                  objects.append(object)
                else:
                  spos.append([])
                  subjects.append([])
                  objects.append([])

          # for m,n, ex in zip(spos, true_spos, all_examples):
          #   print(ex)
          #   print(m, n)
          #   print('='*100)
          tp, fp, fn = calculate_metric_relation(spos, true_spos)
          p, r, f1 = get_p_r_f(tp, fp, fn)
          # print("========metric========")
          # print("precision:{} recall:{} f1:{}".format(p, r, f1))

          return p, r, f1
                
                

    def test(self, model_path):
        model = Casrel(self.args, self.tag2id)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        spos = []
        true_spos = []
        subjects = []
        objects = []
        all_examples = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for dev_batch in dev_batch_data[:-1]:
                    dev_batch = dev_batch.to(device)
                
                seq_output, subject_preds = model.predict_subject(dev_batch_data[0], dev_batch_data[1],dev_batch_data[2])
                # 注意这里需要先获取subject，然后再来获取object和关系，和训练直接使用subject_ids不一样
                cur_batch_size = dev_batch_data[0].shape[0]
                dev_examples = dev_batch_data[-1]
                true_spos += [i[1] for i in dev_examples]
                all_examples += [i[0] for i in dev_examples]
                subject_labels = dev_batch_data[3].cpu().numpy()
                object_labels = dev_batch_data[4].cpu().numpy()
                subject_ids = get_subject_ids(subject_preds, dev_batch_data[1])

                example_lengths = torch.sum(dev_batch_data[1].cpu(), -1)
                
                for i in range(cur_batch_size):
                  seq_output_tmp = seq_output[i, ...]
                  subject_ids_tmp = subject_ids[i]
                  length = example_lengths[i]
                  example = dev_examples[i][0]
                  if subject_ids_tmp:
                    seq_output_tmp = seq_output_tmp.unsqueeze(0).repeat(len(subject_ids_tmp), 1, 1)
                    subject_ids_tmp = torch.tensor(subject_ids_tmp, dtype=torch.long, device=device)
                    if len(seq_output_tmp.shape) == 2:
                      seq_output_tmp = seq_output_tmp.unsqueeze(0)
                    object_preds = model.predict_object([seq_output_tmp, subject_ids_tmp])
                    spo, subject, object = get_spo(object_preds, subject_ids_tmp, length, example, self.id2tag)
                    spos.append(spo)
                    subjects.append(subject)
                    objects.append(object)
                  else:
                    spos.append([])
                    subjects.append([])
                    objects.append([])



            for i, (m,n, ex) in enumerate(zip(spos, true_spos, all_examples)):
              if i <= 10:
                print(ex)
                print(m, n)
                print('='*100)
            tp, fp, fn = calculate_metric_relation(spos, true_spos)
            p, r, f1 = get_p_r_f(tp, fp, fn)
            print("========metric========")
            print("precision:{} recall:{} f1:{}".format(p, r, f1))

            return p, r, f1

    def predict(self, raw_text, model, tokenizer):
        model.eval()
        with torch.no_grad():
            tokens = fine_grade_tokenize(raw_text, tokenizer)
            if len(tokens) > self.args.max_seq_len:
              tokens = tokens[:self.args.max_seq_len]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(token_ids)
            token_type_ids = [0] * len(token_ids)
            if len(token_ids) < self.args.max_seq_len:
              token_ids = token_ids + [0] * (self.args.max_seq_len - len(tokens))
              attention_masks = attention_masks + [0] * (self.args.max_seq_len - len(tokens))
              token_type_ids = token_type_ids + [0] * (self.args.max_seq_len - len(tokens))
            assert len(token_ids) == self.args.max_seq_len
            assert len(attention_masks) == self.args.max_seq_len
            assert len(token_type_ids) == self.args.max_seq_len
            token_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(self.device)
            attention_masks = torch.from_numpy(np.array(attention_masks, dtype=np.uint8)).unsqueeze(0).to(self.device)
            token_type_ids = torch.from_numpy(np.array(token_type_ids)).unsqueeze(0).to(self.device)
            seq_output, subject_preds = model.predict_subject(token_ids, attention_masks, token_type_ids)
            subject_ids = get_subject_ids(subject_preds, attention_masks)

            cur_batch_size = seq_output.shape[0]
            spos = []
            subjects = []
            objects = []
            for i in range(cur_batch_size):
                seq_output_tmp = seq_output[i, ...]
                subject_ids_tmp = subject_ids[i]
                length = len(tokens)
                example = raw_text
                if any(subject_ids_tmp):
                  seq_output_tmp = seq_output_tmp.unsqueeze(0).repeat(len(subject_ids_tmp), 1, 1)

                  subject_ids_tmp = torch.tensor(subject_ids_tmp, dtype=torch.long, device=device)
                  if len(seq_output_tmp.shape) == 2:
                    seq_output_tmp = seq_output_tmp.unsqueeze(0)
                  object_preds = model.predict_object([seq_output_tmp, subject_ids_tmp])

                  spo, subject, object = get_spo(object_preds, subject_ids_tmp, length, example, self.id2tag)

                  subjects.append(subject)
                  objects.append(object)
                  spos.append(spo)
            print("文本：", raw_text)
            print('主体：', subjects)
            print('客体：', objects)
            print('关系：', spos)
            print("="*100)

if __name__ == '__main__':
    data_name = 'ske'
    model_name = 'bert'

    set_logger(os.path.join(args.log_dir, '{}.log'.format(model_name)))
    if data_name == "ske":
        args.data_dir = './data/ske'
        data_path = os.path.join(args.data_dir, 'raw_data')
        label_list = read_json(os.path.join(args.data_dir, 'mid_data'), 'predicates')
        tag2id = {}
        id2tag = {}
        for k,v in enumerate(label_list):
            tag2id[v] = k
            id2tag[k] = v
        
        logger.info(args)
        max_seq_len = args.max_seq_len
        tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext/vocab.txt')

        model = Casrel(args, tag2id)
        model, device = load_model_and_parallel(model, args.gpu_ids)

        collate = data_loader.Collate(max_len=max_seq_len, tag2id=tag2id, device=device, tokenizer=tokenizer)


        train_dataset = data_loader.MyDataset(file_path=os.path.join(data_path, 'train_data.json'), 
                    tokenizer=tokenizer, 
                    max_len=max_seq_len)

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate.collate_fn) 
        dev_dataset = data_loader.MyDataset(file_path=os.path.join(data_path, 'dev_data.json'), 
                    tokenizer=tokenizer, 
                    max_len=max_seq_len)

        dev_dataset = dev_dataset[:args.use_dev_num]
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate.collate_fn) 


        bertForNer = BertForRe(args, train_loader, dev_loader, dev_loader, id2tag, tag2id, model, device)
        # bertForNer.train()

        model_path = './checkpoints/bert/model.pt'.format(model_name)
        # bertForNer.test(model_path)
        
        texts = [
        '查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部',
        '《离开》是由张宇谱曲，演唱',
        '《愤怒的唐僧》由北京吴意波影视文化工作室与优酷电视剧频道联合制作，故事以喜剧元素为主，讲述唐僧与佛祖打牌，得罪了佛祖，被踢下人间再渡九九八十一难的故事',
        '李治即位后，萧淑妃受宠，王皇后为了排挤萧淑妃，答应李治让身在感业寺的武则天续起头发，重新纳入后宫',
        '《工业4.0》是2015年机械工业出版社出版的图书，作者是（德）阿尔冯斯·波特霍夫，恩斯特·安德雷亚斯·哈特曼',
        '周佛海被捕入狱之后，其妻杨淑慧散尽家产请蒋介石枪下留人，于是周佛海从死刑变为无期，不过此人或许作恶多端，改判没多久便病逝于监狱，据悉是心脏病发作',
        '《李烈钧自述》是2011年11月1日人民日报出版社出版的图书，作者是李烈钧',
        '除演艺事业外，李冰冰热心公益，发起并亲自参与多项环保慈善活动，积极投身其中，身体力行担起了回馈社会的责任于02年出演《少年包青天》，进入大家视线',
        '马志舟，1907年出生，陕西三原人，汉族，中国共产党，任红四团第一连连长，1933年逝世',
        '斑刺莺是雀形目、剌嘴莺科的一种动物，分布于澳大利亚和新西兰，包括澳大利亚、新西兰、塔斯马尼亚及其附近的岛屿',
        '《课本上学不到的生物学2》是2013年上海科技教育出版社出版的图书',
        ]
        model = Casrel(args, tag2id)
        model, device = load_model_and_parallel(model, args.gpu_ids, model_path)
        for text in texts:
          bertForNer.predict(text, model, tokenizer)

