import os
import json
import logging
from transformers import BertTokenizer
from utils import common_utils

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels

    def __repr__(self):
        string = ""
        for key, value in self.__dict__.items():
            string += f"{key}: {value}\n"
        return f"<{string}>"

class ReProcessor:
    @staticmethod
    def read_json(file_path):
      pass

    def get_examples(self, raw_examples, set_type):
      pass


class SKEProcessor(ReProcessor):
    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = f.readlines()
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
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
                examples.append(InputExample(set_type=set_type,
                                  text=text,
                                  labels=labels))

        return examples



if __name__ == "__main__":
  skeProcessor = SKEProcessor()
  raw_examples = skeProcessor.read_json('data/ske/raw_data/train_data.json')
  examples = skeProcessor.get_examples(raw_examples, set_type="train")
  for i in range(5):
    print(examples[i])