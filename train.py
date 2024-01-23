import json 
import sys 
import os 
import torch 
from model.model import BiEncoder
from trainer.trainer import Trainer



if __name__ == "__main__": 

    device= torch.device("cuda") 

    model= BiEncoder(model_name = 'vinai/phobert-base-v2', required_grad= True, num_label= 1)

    model.to(device)

    trainer= Trainer(model= model,
                    tokenizer_name= 'vinai/phobert-base-v2',
                    path_datatrain= ['Embedding/data/contexts.json', 'Embedding/data/questions.json', 'Embedding/data/train_triples_ids.csv'], 
                    path_dataeval= None,
                    batch_size= 2, 
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
                    path_ckpt_step= 'test.pt', 
                    device= device,
                    use_wandb=False,
                    loss= 'triplet', 
                )
    trainer.fit(step_save= 1000,
                path_ckpt_epoch= 'epochs.pt',
                logging_step = 10)