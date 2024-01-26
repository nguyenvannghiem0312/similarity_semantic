from typing import Type, Optional, List
import os
import json

import torch 
import torch.nn as nn
from transformers import AutoModel

class BiEncoder(torch.nn.Module):
    def __init__(self, model_name = 'vinai/phobert-base-v2',
                 required_grad = True,
                 pooler: str = "mean") :
        super(BiEncoder, self).__init__()

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states= True)

        if not required_grad:
            self.model.required_grad(False)

        self.pooler = pooler

    def save_model(self, path):
        pooler_config = {
            'pooler': self.pooler
        }
        with open(path + '/pooler_config.json', 'w+') as config_file:
            json.dump(pooler_config, config_file, indent=2)

        return self.model.save_pretrained(path)
    
    def get_embedding(self, inputs):
        embedding_model = self.model(**inputs, output_hidden_states=True, return_dict=True)

        last_hidden = embedding_model.last_hidden_state
        pooler_output = embedding_model.pooler_output
        # hidden_states = embedding_model.hidden_states
        
        if self.pooler == "cls":
            return pooler_output
        if self.pooler == "mean":
            return ((last_hidden * inputs['attention_mask'].unsqueeze(-1)).sum(1) / inputs['attention_mask'].sum(-1).unsqueeze(-1))
    
