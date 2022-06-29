import torch
import torch.nn as nn
from transformers import BertModel


class MyBCELoss(nn.BCELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, inputs, targets):
        # subject_preds: torch.Size([16, 256, 2])
        # subject_labels: torch.Size([16, 256, 2])
        # object_labels: torch.Size([16, 256, 49, 2])
        # object_preds: torch.Size([16, 256, 49, 2])
        subject_preds, object_preds = inputs
        subject_labels, object_labels, mask = targets
        # sujuect部分loss
        subject_loss = super().forward(subject_preds, subject_labels)
        subject_loss = subject_loss.mean(dim=-1)
        subject_loss = (subject_loss * mask).sum() / mask.sum()
        # object部分loss
        object_loss = super().forward(object_preds, object_labels)
        object_loss = object_loss.mean(dim=-1).sum(dim=-1)
        object_loss = (object_loss * mask).sum() / mask.sum()
        return subject_loss + object_loss


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, conditional_size=False, weight=True, bias=True, norm_mode='normal', **kwargs):
        """layernorm 层，这里自行实现，目的是为了兼容 conditianal layernorm，使得可以做条件文本生成、条件分类等任务
           条件layernorm来自于苏剑林的想法，详情：https://spaces.ac.cn/archives/7124
        """
        super(LayerNorm, self).__init__()
        
        # 兼容roformer_v2不包含weight
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        # 兼容t5不包含bias项, 和t5使用的RMSnorm
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.norm_mode = norm_mode

        self.eps = eps
        self.conditional_size = conditional_size
        if conditional_size:
            # 条件layernorm, 用于条件文本生成,
            # 这里采用全零初始化, 目的是在初始状态不干扰原来的预训练权重
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)

    def forward(self, x):
        inputs = x[0]  # 这里是visible_hiddens

        if self.norm_mode == 'rmsnorm':
            # t5使用的是RMSnorm
            variance = inputs.to(torch.float32).pow(2).mean(-1, keepdim=True)
            o = inputs * torch.rsqrt(variance + self.eps)
        else:
            # 归一化是针对于inputs
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            o = (inputs - u) / torch.sqrt(s + self.eps)

        if not hasattr(self, 'weight'):
            self.weight = 1
        if not hasattr(self, 'bias'):
            self.bias = 0

        if self.conditional_size:
            cond = x[1]  # 这里是repeat_hiddens
            # 三者的形状都是一致的
            # print(inputs.shape, cond.shape, o.shape)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            
            return (self.weight + self.dense1(cond)) * o + (self.bias + self.dense2(cond))
        else:
            return self.weight * o + self.bias


# 定义bert上的模型结构
class Casrel(nn.Module):
    def __init__(self, args, tag2id):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.tag2id = tag2id
        self.linear1 = nn.Linear(768, 2)
        # 768*2
        self.condLayerNorm = LayerNorm(hidden_size=768, conditional_size=768*2)
        self.linear2 = nn.Linear(768, len(tag2id)*2)
        self.crierion = MyBCELoss()

    @staticmethod
    def extract_subject(inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        output, subject_ids = inputs
        start = torch.gather(output, dim=1, index=subject_ids[:, :1].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        end = torch.gather(output, dim=1, index=subject_ids[:, 1:].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        subject = torch.cat([start, end], 2)
        # print(subject.shape)
        return subject[:, 0]

    def forward(self, 
          token_ids, 
          attention_masks, 
          token_type_ids,
          subject_labels=None,
          object_labels=None,
          subject_ids=None):
        # 预测subject
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )  
        seq_output = bert_outputs[0]  # [btz, seq_len, hdsz]
        subject_preds = (torch.sigmoid(self.linear1(seq_output)))**2  # [btz, seq_len, 2]

        # 传入subject，预测object
        # 通过Conditional Layer Normalization将subject融入到object的预测中
        # 理论上应该用LayerNorm前的，但是这样只能返回各个block顶层输出，这里和keras实现不一致
        subject = self.extract_subject([seq_output, subject_ids])
        output = self.condLayerNorm([seq_output, subject])
        output = (torch.sigmoid(self.linear2(output)))**4
        object_preds = output.reshape(*output.shape[:2], len(self.tag2id), 2)
        # print(object_preds.shape, object_labels.shape)
        loss = self.crierion([subject_preds, object_preds], [subject_labels, object_labels, attention_masks])
        return loss
        
    def predict_subject(self, token_ids, attention_masks, token_type_ids):
        self.eval()
        with torch.no_grad():
            bert_outputs = self.bert(
                input_ids=token_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids
            )  
            seq_output = bert_outputs[0]  # [btz, seq_len, hdsz]
            subject_preds = (torch.sigmoid(self.linear1(seq_output)))**2  # [btz, seq_len, 2]
        return seq_output, subject_preds
    
    def predict_object(self, inputs):
        self.eval()
        with torch.no_grad():
            seq_output, subject_ids = inputs
            subject = self.extract_subject([seq_output, subject_ids])
            output = self.condLayerNorm([seq_output, subject])
            output = (torch.sigmoid(self.linear2(output)))**4
            object_preds = output.reshape(*output.shape[:2], len(self.tag2id), 2)
        return object_preds