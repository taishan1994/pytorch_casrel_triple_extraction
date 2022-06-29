import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.common_utils import sequence_padding


class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path):
        return file_path



# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        examples = []
        with open(filename, encoding='utf-8') as f:
            raw_examples = f.readlines()
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
            # print(i,item)
            item = json.loads(item)
            text = item['text']
            spo_list = item['spo_list']
            labels = [] # [subject, predicate, object]
            for spo in spo_list:
                subject = spo['subject']
                object = spo['object']
                predicate = spo['predicate']
                labels.append([subject, predicate, object])
            examples.append((text, labels))
        return examples

class Collate:
  def __init__(self, max_len, tag2id, device, tokenizer):
      self.maxlen = max_len
      self.tag2id = tag2id
      self.id2tag = {v:k for k,v in tag2id.items()}
      self.device = device
      self.tokenizer = tokenizer

  def collate_fn(self, batch):
      def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
      batch_subject_labels = []
      batch_object_labels = []
      batch_subject_ids = []
      batch_token_ids = []
      batch_attention_mask = []
      batch_token_type_ids = []
      callback = []
      for i, (text, text_labels) in enumerate(batch):
          if len(text) > self.maxlen:
            text = text[:self.maxlen]
          tokens = [i for i in text]
          spoes = {}
          callback_text_labels = []
          for s, p, o in text_labels:
            p = self.tag2id[p]
            s_idx = search(s, text)
            o_idx = search(o, text)
            if s_idx != -1 and o_idx != -1:
              callback_text_labels.append((s, self.id2tag[p], o))
              s = (s_idx, s_idx + len(s) - 1)
              o = (o_idx, o_idx + len(o) - 1, p)
              if s not in spoes:
                  spoes[s] = []
              spoes[s].append(o)
          # print(text_labels)
          # print(text)
          # print(spoes)
          if spoes:
            # subject标签
            subject_labels = np.zeros((len(tokens), 2))
            for s in spoes:
                subject_labels[s[0], 0] = 1  # subject首
                subject_labels[s[1], 1] = 1  # subject尾
            start, end = np.array(list(spoes.keys())).T  
            start = np.random.choice(start)
            end = np.random.choice(end[end >= start])
            # 这里取出的可能不是一个真实的subject
            subject_ids = (start, end)  
            # 对应的object标签
            object_labels = np.zeros((len(tokens), len(self.tag2id), 2))
            for o in spoes.get(subject_ids, []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1
            # 构建batch
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            batch_token_ids.append(token_ids)  # 前面已经限制了长度
            batch_attention_mask.append([1] * len(token_ids))
            batch_token_type_ids.append([0] * len(token_ids))
            batch_subject_labels.append(subject_labels)
            batch_object_labels.append(object_labels)
            batch_subject_ids.append(subject_ids)
            callback.append((text, callback_text_labels))
      batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=self.maxlen), dtype=torch.long, device=self.device)
      attention_mask = torch.tensor(sequence_padding(batch_attention_mask, length=self.maxlen), dtype=torch.long, device=self.device)
      token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=self.maxlen), dtype=torch.long, device=self.device)
      batch_subject_labels = torch.tensor(sequence_padding(batch_subject_labels, length=self.maxlen), dtype=torch.float, device=self.device)
      batch_object_labels = torch.tensor(sequence_padding(batch_object_labels, length=self.maxlen), dtype=torch.float, device=self.device)
      batch_subject_ids = torch.tensor(batch_subject_ids, dtype=torch.long, device=self.device)

      return batch_token_ids, attention_mask, token_type_ids, batch_subject_labels, batch_object_labels, batch_subject_ids, callback


if __name__ == "__main__":
  from transformers import BertTokenizer
  max_len = 256
  tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext/vocab.txt')
  train_dataset = MyDataset(file_path='data/ske/raw_data/train_data.json', 
              tokenizer=tokenizer, 
              max_len=max_len)
  # print(train_dataset[0])

  with open('data/ske/mid_data/predicates.json') as fp:
    labels = json.load(fp)
  id2tag = {}
  tag2id = {}
  for i,label in enumerate(labels):
    id2tag[i] = label
    tag2id[label] = i
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  collate = Collate(max_len=max_len, tag2id=tag2id, device=device, tokenizer=tokenizer)
  # collate.collate_fn(train_dataset[:20])
  batch_size = 2
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn) 

  for i, batch in enumerate(train_dataloader):
    print(batch)
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2].shape)
    print(batch[3].shape)
    print(batch[4].shape)
    print(batch[5].shape)
    print(batch[6])
    break