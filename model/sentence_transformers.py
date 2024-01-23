import torch 
from typing import Type, Optional, List
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .pooling import Pooling
from .model import BiEncoder

class SentenceTransformer: 
    def __init__(self, model_name= 'vinai/phobert-base-v2', required_grad= True, num_label= 1,
                 dropout= 0.1, hidden_dim= 256, torch_dtype= torch.float16, device= None):
    
        self.model= BiEncoder(model_name, required_grad, dropout, hidden_dim, num_label)
        self.tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast= True, add_prefix_space= True)
        self.device= device 
        self.torch_dtype= torch_dtype
    
    def load_ckpt(self, path): 
        self.model.load_state_dict(torch.load(path, map_location= 'cpu')['model_state_dict'])
        self.model.to(self.device, dtype= self.torch_dtype)

    def _preprocess(self): 
        if self.model.training: 
            self.model.eval() 
    
    def _preprocess_tokenize(self, text): 
        inputs= self.tokenizer.batch_encode_plus(text, return_tensors= 'pt', 
                            padding= 'max_length', max_length= 256, truncation= True)
        
        return inputs
    
    def encode(self, text: List[str]): 
        self._preprocess()

        inputs= self._preprocess_tokenize(text)

        with torch.no_grad(): 
            embedding= self.model.get_embedding(dict( (i, j.to(self.device)) for i,j in inputs.items()))
                
        return torch.tensor(embedding) 