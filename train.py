import json 
import sys 
import os 

from torch.utils.data import DataLoader
import torch 

from model.model import BiEncoder
from trainer.trainer import Trainer
from loaders.datasets import SemanticDataset
from loaders.dataloader import DataTokenizer

if __name__ == "__main__": 

    device= torch.device("cuda") 

    model_name = 'vinai/phobert-base-v2'
    model= BiEncoder(model_name = model_name, required_grad= True, pooler = "cls")
    type_format = 'A'
    model.to(device)

    path_datatrain= ['Embedding/data/contexts.json', 'Embedding/data/questions.json', 'Embedding/data/train_triples_ids.csv']
    data_train = SemanticDataset(path_datatrain[0], path_datatrain[1], path_datatrain[2], type_format=type_format)

    path_dataeval= ['Embedding/data/contexts.json', 'Embedding/data/questions.json', 'Embedding/data/test_triples_ids.csv']
    data_eval = SemanticDataset(path_dataeval[0], path_dataeval[1], path_dataeval[2], type_format=type_format)
    batch_size = 2
    
    tokenizer= DataTokenizer(tokenizer_model=model_name)
    # dataloader
    dataloader_train= DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True
    )

    if data_eval != None:
        dataloader_eval= DataLoader(
            data_eval,
            batch_size=batch_size
        )

    trainer= Trainer(model= model,
                    tokenizer= tokenizer,
                    dataloader_train=dataloader_train, 
                    dataloader_eval=dataloader_eval,
                    shuffle= True, 
                    num_workers= 16,
                    pin_memory= True, 
                    prefetch_factor= 8, 
                    persistent_workers= True, 
                    gradient_accumlation_steps= 4,
                    learning_rate= 3e-4, 
                    weight_decay= 0.1, 
                    eps= 1e-6, 
                    warmup_steps= 150, 
                    epochs= 5,
                    device= device,
                    use_wandb=False,
                    loss= 'cosine_similarity', 
                    type_format=type_format
                )
    trainer.fit(step_save= 1000,
                logging_step = 10)