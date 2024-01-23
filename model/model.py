import torch 
from typing import Type, Optional, List
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .pooling import Pooling
from Embedding.loaders.dataloader import DataTokenizer

class BiEncoder(torch.nn.Module):
    def __init__(self, model_name = 'vinai/phobert-base-v2',
                 required_grad = True,
                 dropout = 0.1,
                 hidden_dim = 256,
                 max_length = 256,
                 num_label=None,
                 device="cuda") :
        super(BiEncoder, self).__init__()

        self.device = device
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states= True)
        self.tokenizer= DataTokenizer(model_name, max_length=self.max_length)

        if not required_grad:
            self.model.required_grad(False)

        self.pooler = Pooling(method= "mean")
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)


        if not num_label: 
            self.fnn= nn.Linear(hidden_dim * 2 + 1, 128)
        else:
            self.fnn= nn.Linear(hidden_dim * 2 + 1, num_label)

    def load_ckpt(self, path): 
        self.model.load_state_dict(torch.load(path, map_location= 'cpu')['model_state_dict'])
        self.model.to(self.device)

    
    def encode(self, text: List[str]): 
        self.model.eval()
        inputs= self.tokenizer._tokenizer(text)

        with torch.no_grad(): 
            embedding= self.model.get_embedding(dict( (i, j.to(self.device)) for i,j in inputs.items()))
                
        return torch.tensor(embedding) 
    
    def get_embedding(self, inputs):
        embedding_model = self.model(**inputs)
        embedding_pooler = self.pooler(embedding_model.hidden_states[0])

        return self.dropout1(embedding_pooler)
    
    def forward(self, inputs_1, inputs_2):
        emb_1 = self.get_embedding(inputs_1)
        emb_2 = self.get_embedding(inputs_2)

        x = torch.concat((emb_1, emb_2, torch.norm(emb_2 - emb_1, p= 2, dim= -1).view(-1, 1)), dim= -1)
        x = self.dropout2(x)
        x = self.fnn(x)

        return x 

