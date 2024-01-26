from typing import Type, Optional, List
import json

import torch 
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from .model import BiEncoder

class SentenceTransformer: 
    def __init__(self, model_name= 'vinai/phobert-base-v2', model_pretrained=None, required_grad= True, device = "cuda"):
    
        self.model_name = model_name
        self.model_pretrained = model_pretrained
        self.device = device
        if model_pretrained != None:
            with open(model_pretrained + '/pooler_config.json', 'r') as config_file:
                self.pooler_config = json.load(config_file)
                self.pooler = self.pooler_config['pooler']

            self.model= BiEncoder(model_pretrained, required_grad, pooler=self.pooler)
            self.tokenizer= AutoTokenizer.from_pretrained(model_pretrained, use_fast= True, add_prefix_space= True)
        else:
            self.model= BiEncoder(model_name, required_grad)
            self.tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast= True, add_prefix_space= True)
        self.model.to(device)
        self.scores = []
        
    
    def load_ckpt(self, path): 
        self.model.load_state_dict(torch.load(path, map_location= 'cpu')['model_state_dict'])
        self.model.to(self.device)

    def _preprocess(self): 
        if self.model.training: 
            self.model.eval() 
    
    def _preprocess_tokenize(self, text): 
        inputs= self.tokenizer.batch_encode_plus(text, return_tensors= 'pt', 
                            padding= 'max_length', max_length= 256, truncation= True)
        
        return inputs
    
    def encode_pair(self, text_1: List[str], text_2: List[str]): 
        self._preprocess()

        embedding_1= self.encode(text_1)
        embedding_2= self.encode(text_2)
        
        self.scores = self.calculate_cosine_similarity(embeddings1=embedding_1, embeddings2=embedding_2)
        return torch.tensor(embedding_1), torch.tensor(embedding_2) 
    
    def get_score(self):
        return self.scores
    
    def encode(self, text: List[str]): 
        self._preprocess()

        inputs= self._preprocess_tokenize(text).to(self.device)
        with torch.no_grad(): 
            # for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            #     inputs[key] = inputs[key].to(self.device)
            embedding= self.model.get_embedding(inputs)
                
        return torch.tensor(embedding) 

    def calculate_cosine_similarity(self, embeddings1, embeddings2):
        assert len(embeddings1) == len(embeddings2), "Error shape"

        embeddings1_tensor_normalized = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2_tensor_normalized = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

        similarities = cosine_similarity(embeddings1_tensor_normalized, embeddings2_tensor_normalized, dim=1)

        return similarities
    
    
    
       