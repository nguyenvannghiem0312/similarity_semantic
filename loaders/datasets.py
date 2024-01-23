from typing import Any
import pandas as pd 
import torch 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loaders.datareader import DataReader 

class TripletDataset(Dataset): 
    def __init__(self, 
                 path_data_contexts: str,
                 path_data_questions: str,
                 path_data_triples_ids: str,
                ):
        self.contexts = DataReader(path_data_contexts).read()
        self.questions = DataReader(path_data_questions).read()
        self.triples_ids = DataReader(path_data_triples_ids).read()

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
    
    def __call__(self, type_format = 'A'):
        if type_format == 'A': 
            return self.convert_to_format_text_text_label()
        if type_format == 'B':
            return self.convert_to_text_positive_negative()
        assert type_format != 'A' and type_format != 'B', "Type format must be: 'A' or 'B'"
    

# def main():
#     contexts_file_path = 'Embedding/data/contexts.json'
#     questions_file_path = 'Embedding/data/questions.json'
#     triples_ids_file_path = 'Embedding/data/train_triples_ids.csv'
#     data_loader = ConvertFormatDataset(contexts_file_path, questions_file_path, triples_ids_file_path)
#     data = data_loader(type_format='B')
#     print(data)
# main()
    