from typing import Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DataTokenizer:
    def __init__(self,
                 tokenizer_model = 'vinai/phobert-base-v2',
                 max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length

    def _tokenizer(self, text):
        return self.tokenizer(text, truncation=True, padding='max_length', return_tensors='pt', max_length=self.max_length)
    
    def _max_length(self):
        return self.max_length
    
    def __call__(self, data, type_format='A'):
        if type_format == 'A':
            text1, text2, label = data['text1'], data['text2'], data['label']
            return {
                't_1': self._tokenizer(text1),
                't_2': self._tokenizer(text2),
                'label': label.clone().detach()
            }
        if type_format == 'B':
            text, positive, negative = data['text'], data['positive'], data['negative']
            return {
                't': self._tokenizer(text),
                'pos': self._tokenizer(positive),
                'neg': self._tokenizer(negative)
            }
            