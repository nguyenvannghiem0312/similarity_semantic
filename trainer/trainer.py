from typing import Type, Optional, List
import os
import json

import torch
from torch.cuda.amp import autocast, GradScaler
from transformers.optimization import get_cosine_schedule_with_warmup
import wandb

from losses.ContrastiveLoss import ContrastiveLoss
from losses.SimilarityLoss import CosineSimilarityLoss
from losses.TripletLoss import TripletLoss
from callbacks.EarlyStopping import EarlyStopping

# from evaluation.BinaryClassification import BinaryClassificationEvaluator


class Trainer:
    def __init__(self,
        model, 
        tokenizer, 
        dataloader_train,
        dataloader_eval,
        device: Type[torch.device], 
        # batch_size: int = 8, 
        shuffle: Optional[bool]= True, 
        num_workers: int= 16, 
        pin_memory: Optional[bool]= True, 
        prefetch_factor: int= 8, 
        persistent_workers: Optional[bool]= True, 
        gradient_accumlation_steps: int= 16, 
        learning_rate: float= 1e-4, 
        weight_decay: Optional[float]= 0.1, 
        eps: Optional[float]= 1e-6, 
        warmup_steps: int= 150, 
        epochs: Optional[int]= 1, 
        loss: Optional[str]='triplet',
        use_wandb: bool= True,
        early_stop: int=5,
        type_format = 'A'
        ):

        # base 
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval
        self.type_format = type_format
        # dataset
        # self.type_format = type_format
        # self.data_train = SemanticDataset(path_datatrain[0], path_datatrain[1], path_datatrain[2], type_format=self.type_format)
        # # self.data_train = self.data_train()

        # self.path_dataeval = path_dataeval
        # self.data_eval = None
        # if len(self.path_dataeval) > 0:
        #     self.data_eval = SemanticDataset(path_dataeval[0], path_dataeval[1], path_dataeval[2], type_format=self.type_format)
            # self.data_eval = self.data_eval()

        # train args
        # self.batch_size= batch_size
        self.shuffle= shuffle
        self.num_workers= num_workers
        self.pin_memory= pin_memory
        self.prefetch_factor= prefetch_factor
        self.persistent_workers= persistent_workers 
        self.grad_accum= gradient_accumlation_steps 
        self.lr= learning_rate 
        self.weight_decay= weight_decay 
        self.eps= eps 
        self.warmup_steps= warmup_steps
        self.epochs= epochs
        
        # device
        self.device= device 

    
        # mixer precision 
        self.scaler= GradScaler()

        # set loss
        self.loss = loss
        if loss == 'triplet':
            self.criterion = TripletLoss()
        elif loss == 'cosine_similarity':
            self.criterion = CosineSimilarityLoss()
        elif loss == 'contrastive':
            self.criterion = ContrastiveLoss()

        # set optimizer
        self.optimizer= torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay= self.weight_decay, 
                                      eps= self.eps)
        step_epoch = len(self.dataloader_train)
        self.total_steps= int(step_epoch / self.grad_accum) * self.epochs

        self.scheduler= get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps= self.warmup_steps,
                                                        num_training_steps= self.total_steps)
        # early stopping
        self.early_stop = EarlyStopping(patience=early_stop)

        # other 
        self.use_wandb= use_wandb 
        self.global_step = 0
    
    def compute_loss(self, data):

        if self.type_format=='A':
            label = data['label'].to(self.device)
            embed_t1, embed_t2 = None, None
            if self.loss == 'cosine_similarity' or self.loss == 'contrastive':
                for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    data['t_1'][key] = torch.squeeze(data['t_1'][key], dim=1).to(self.device)
                    data['t_2'][key] = torch.squeeze(data['t_2'][key], dim=1).to(self.device)
                embed_t1= self.model.get_embedding(data['t_1'])
                embed_t2= self.model.get_embedding(data['t_2'])
            return self.criterion(embed_t1, embed_t2, label.to(dtype= torch.float32))
        

        if self.type_format=='B':
            if self.loss == 'triplet':
                for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    data['t'][key] = torch.squeeze(data['t'][key], dim=1).to(self.device)
                    data['pos'][key] = torch.squeeze(data['pos'][key], dim=1).to(self.device)
                    data['neg'][key] = torch.squeeze(data['neg'][key], dim=1).to(self.device)
                embed_t= self.model.get_embedding(data['t'])
                embed_pos= self.model.get_embedding(data['pos'])
                embed_neg= self.model.get_embedding(data['neg'])
            return self.criterion(embed_t, embed_pos, embed_neg)
        
    def _train_on_epoch(self, index_grad, logging_step: Optional[int]= 100, step_save: Optional[int]= 1000, save_directory: Type[str]= 'model'): 
        self.model.train()
        total_loss, total_count= 0, 0
        step_loss, step_fr= 0, 0

        for idx, data in enumerate(self.dataloader_train): 
            with autocast():
                loss= self.compute_loss(data)
                loss /= self.grad_accum
            # print(loss)
            self.scaler.scale(loss).backward()

            step_loss += loss.item() * self.grad_accum
            step_fr +=1 

            if ((idx + 1) % self.grad_accum == 0) or (idx + 1 ==len(self.dataloader_train)): 
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.) 
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none= True)
                self.scheduler.step() 

                if self.use_wandb:
                    wandb.log({"Train loss": total_loss / total_count})
            
            total_loss += loss.item() 
            total_count += 1 

            self.global_step += 1
            if (total_count + 1) % (logging_step * self.grad_accum) == 0: 
                print(f'Step: [{(total_count + 1 )/ (self.grad_accum):.0f}/{self.total_steps / self.epochs:.0f}], Loss: {(total_loss / total_count) * self.grad_accum}')
                step_loss = 0 
                step_fr = 0 
                index_grad[0] += 1 

            if (self.global_step + 1) % (step_save * self.grad_accum) ==0:
                self.save_model(save_directory=save_directory + f'/{self.global_step + 1}')
                # torch.save({'step': idx + 1,
                #             'model_state_dict': self.model.state_dict(),
                #             'optimizer_state_dict': self.optimizer.state_dict(),
                #             'scheduler': self.scheduler.state_dict(),
                #                 },  self.ckpt_step)
        
        return (total_loss / total_count) * self.grad_accum
    
    def _evaluate(self): 
        self.model.eval()
        total_loss, total_count= 0, 0
        with torch.no_grad(): 
            for idx, data in enumerate(self.dataloader_eval): 
                loss= self.compute_loss(data)
                total_loss += loss.item() 
                if self.use_wandb: 
                    wandb.log({"Eval loss": loss.item()})
                total_count += 1 
        
        return total_loss / total_count 

    def fit(self, logging_step:Type[int]= 100, step_save: Type[int]= 1000, save_directory: Type[str]= 'model'):
        print(' START ')
        index_grad= [1] 
        log_loss = 1e9
        for epoch in range(1, self.epochs + 1): 
            print(f' EPOCH {epoch} ')
            train_loss= self._train_on_epoch(index_grad, logging_step, step_save, save_directory)
            print(f'End of epoch {epoch} - loss: {train_loss}')            
            if self.dataloader_eval != None:
                print(' EVALUATE ')
                val_loss= self._evaluate()
                print(f'Evaluate loss: {val_loss}')
                if self.early_stop(val_loss=val_loss) == True:
                    self.save_model(save_directory=save_directory + '/early_stopping')
                
                if val_loss < log_loss: 
                    log_loss = val_loss
                    print(f'Saving checkpoint have best {log_loss}')
                    self.save_model(save_directory=save_directory + f'/epochs{epoch}')
                    # torch.save({'epoch': epoch, 
                    #             'model_state_dict': self.model.state_dict(), 
                    #             'scheduler': self.scheduler.state_dict()}, 
                    #             path_ckpt_epoch)

            # if train_loss < log_loss: 
            #     log_loss = train_loss
            #     print(f'Saving checkpoint have best {log_loss}')
            #     torch.save({'epoch': epoch, 
            #                 'model_state_dict': self.model.state_dict(), 
            #                 'scheduler': self.scheduler.state_dict()},
            #                 path_ckpt_epoch)
        self.save_model(save_directory=save_directory + '/model')
                
    def save_model(self, save_directory: Type[str]= 'model'):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.model.save_model(save_directory)
        self.tokenizer.save_tok(save_directory)

        print(f"Model and tokenizer saved to {save_directory}")
    
        