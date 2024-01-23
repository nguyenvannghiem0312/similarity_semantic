import os
import random
import numpy as np
import torch
from angle_emb import AnglE, AngleDataTokenizer
from datasets import load_dataset, Dataset
import json
import pandas as pd
from typing import Any, Dict, Optional, List, Union, Tuple, Callable
import csv

class DataLoader:
    def __init__(self, 
                 contexts_file: str, 
                 questions_file: str, 
                 triples_ids_file: str):
        self.contexts = self.load_json(contexts_file)
        self.questions = self.load_json(questions_file)
        self.triples_ids = self.load_csv(triples_ids_file)

    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def load_csv(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        return data

    def convert_to_text_positive_negative(self):
        results = []
        for query_id, pos_id, neg_id in self.triples_ids:
            text = self.questions.get(query_id, "")
            positive = self.contexts.get(pos_id, "")
            negative = self.contexts.get(neg_id, "")
            results.append({'text': text, 'positive': positive, 'negative': negative})
        return results
    
    def train_test_split(self, data, train_ratio = 0.8):
        num_train = int((len(data) * train_ratio))
        return data[:num_train], data[num_train:]

class EmbeddingTrainer:
    def __init__(self,
                 model_name_or_path: str,
                 max_length: int = 512,
                 pretrained_model_path: Optional[str] = None,
                 pooling_strategy: str = 'cls',):
        self.model_name_or_path = model_name_or_path
        self.pretrained_model_path = pretrained_model_path
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.angle = AnglE.from_pretrained(self.model_name_or_path,
                                    max_length=self.max_length, 
                                    pooling_strategy=self.pooling_strategy).cuda()


    def data_tokenizer(self, data, shuffle = False):
        if shuffle == True:
            data = data.shuffle()
        return data.map(AngleDataTokenizer(self.angle.tokenizer, self.angle.max_length))
    
    def __call__(self,
            train_ds: Dataset,
            valid_ds: Optional[Dataset] = None,
            batch_size: int = 32,
            output_dir: Optional[str] = None,
            epochs: int = 1,
            learning_rate: float = 1e-5,
            warmup_steps: int = 1000,
            logging_steps: int = 10,
            eval_steps: Optional[int] = None,
            save_steps: int = 100,
            save_strategy: str = 'steps',
            save_total_limit: int = 10,
            gradient_accumulation_steps: int = 1,
            fp16: bool = False,
            loss_kwargs: Optional[Dict] = None):
        
        train_tokenizer = self.data_tokenizer(train_ds, True)
        valid_tokenizer = self.data_tokenizer(valid_ds)
        self.angle.fit(
            train_ds=train_tokenizer,
            valid_ds=valid_tokenizer,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            loss_kwargs=loss_kwargs,
            fp16=fp16,
        )



def main():
    # Example usage
    contexts_file_path = 'Embedding/data/contexts.json'
    questions_file_path = 'Embedding/data/questions.json'
    triples_ids_file_path = 'Embedding/data/train_triples_ids.csv'
    data_loader = DataLoader(contexts_file_path, questions_file_path, triples_ids_file_path)
    formatted_data = data_loader.convert_to_text_positive_negative()
    # Printing the formatted data
    # for item in formatted_data:
    #     print(item)
    train_ds, val_ds = data_loader.train_test_split(formatted_data, train_ratio=0.8)
    # angle = AnglE.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', max_length=128, pooling_strategy='cls').cuda()

    model = EmbeddingTrainer(model_name_or_path='bert-base-multilingual-cased')
    loss_kwargs= {
                'w1': 0,
                'w2': 5.0,
                'w3': 1.0,
                'cosine_tau': 0,
                'ibn_tau': 20,
                'angle_tau': 1.0
            }
    model(train_ds=Dataset.from_pandas(pd.DataFrame(train_ds)),
              valid_ds=Dataset.from_pandas(pd.DataFrame(val_ds)),
              batch_size=4,
              output_dir='test/',
              loss_kwargs=loss_kwargs)
    # model

# if __name__ == "__main__":
#     main()
main()

