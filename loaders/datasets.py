from typing import Any
import pandas as pd 
import torch 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loaders.datareader import DataReader 

class SemanticDataset(Dataset): 
    def __init__(self, 
                 path_data_contexts: str,
                 path_data_questions: str,
                 path_data_triples_ids: str,
                 type_format = 'A',
                 tokenizer_model = 'vinai/phobert-base-v2',
                 max_length=256
                ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length

        self.contexts = DataReader(path_data_contexts).read()
        self.questions = DataReader(path_data_questions).read()
        self.triples_ids = DataReader(path_data_triples_ids).read()
        
        self.type_format = type_format
        self.data = None
        
        if self.type_format == 'A':
            self.data = self.convert_to_format_text_text_label()
        elif self.type_format == 'B':
            self.data = self.convert_to_text_positive_negative()

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        if self.type_format == 'A':
            text1, text2, label = self.data[index]['text1'], self.data[index]['text2'], self.data[index]['label']
            return {
                't_1': self._tokenizer(text1),
                't_2': self._tokenizer(text2),
                'label': torch.tensor(label)
            }
        if self.type_format == 'B':
            text, positive, negative = self.data[index]['text'], self.data[index]['positive'], self.data[index]['negative']
            return {
                't': self._tokenizer(text),
                'pos': self._tokenizer(positive),
                'neg': self._tokenizer(negative)
            }
        return super().__getitem__(index)
    
    def _tokenizer(self, text):
        return self.tokenizer(text, truncation=True, padding='max_length', return_tensors='pt', max_length=self.max_length)
    
    def _max_length(self):
        return self.max_length
        
    def convert_to_format_text_text_label(self):
        results = []
        for query_id, pos_id, neg_id in self.triples_ids:
            text = self.questions.get(query_id, "")
            positive = self.contexts.get(pos_id, "")
            negative = self.contexts.get(neg_id, "")
            results.append({'text1': text, 'text2': positive, 'label': 1})
            results.append({'text1': text, 'text2': negative, 'label': 0})
        return results

    def convert_to_text_positive_negative(self):
        results = []
        for query_id, pos_id, neg_id in self.triples_ids:
            text = self.questions.get(query_id, "")
            positive = self.contexts.get(pos_id, "")
            negative = self.contexts.get(neg_id, "")
            results.append({'text': text, 'positive': positive, 'negative': negative})
        return results
    
    def __call__(self):
        if self.type_format == 'A':
            self.data = self.convert_to_format_text_text_label()
        elif self.type_format == 'B':
            self.data = self.convert_to_text_positive_negative()
        return self.data
    

