import os 
from typing import Type, Optional, List
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler 
from transformers.optimization import get_cosine_schedule_with_warmup
import wandb 

from loaders.dataloader import DataTokenizer
from loaders.datasets import SemanticDataset

from losses.ContrastiveLoss import ContrastiveLoss
from losses.SimilarityLoss import CosineSimilarityLoss
from losses.TripletLoss import TripletLoss

# from evaluation.BinaryClassification import BinaryClassificationEvaluator

class Trainer: 
    def __init__(self, 
        model, 
        tokenizer_name: Type[str], 
        path_datatrain: List[str],
        device: Type[torch.device], 
        path_dataeval: List[str]= None,
        batch_size: int = 8, 
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
        path_ckpt_step: Optional[str]= 'checkpoint.pt',
        use_wandb: bool= True,
        type_format = 'A'
        ):

        # base 
        self.model = model
        self.tokenizer= DataTokenizer(tokenizer_model=tokenizer_name)
        
        # dataset
        self.type_format = type_format
        self.data_train = SemanticDataset(path_datatrain[0], path_datatrain[1], path_datatrain[2], type_format=self.type_format)
        self.data_train = self.data_train()

        self.path_dataeval = path_dataeval
        self.data_eval = None
        if len(self.path_dataeval) > 0:
            self.data_eval = SemanticDataset(path_dataeval[0], path_dataeval[1], path_dataeval[2], type_format=self.type_format)
            self.data_eval = self.data_eval()

        # train args
        self.batch_size= batch_size
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

        # dataloader
        self.dataloader_train= DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        if self.data_eval != None:
            self.dataloader_eval= DataLoader(
                self.data_eval,
                batch_size=self.batch_size
            )

        # mixer precision 
        self.scaler= GradScaler()

        # path ckpt 
        self.ckpt_step= path_ckpt_step

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
        
        # other 
        self.use_wandb= use_wandb 
    
    def compute_loss(self, data):
        if self.type_format=='A':
            label = data['label'].to(self.device)
            embed_t1, embed_t2 = None, None
            if self.loss == 'cosine_similarity' or self.loss == 'contrastive':
                data = self.tokenizer(data, type_format=self.type_format)
                embed_t1= self.model.get_embedding(
                    dict((i, j.to(self.device)) for i, j in data['t_1'].items() if i in ['input_ids', 'attention_mask'])
                )
                embed_t2= self.model.get_embedding(
                    dict((i, j.to(self.device)) for i, j in data['t_2'].items() if i in ['input_ids', 'attention_mask'])
                )
            return self.criterion(embed_t1, embed_t2, label.to(dtype= torch.float32))
        if self.type_format=='B':
            if self.loss == 'triplet':
                data = self.tokenizer(data, type_format=self.type_format)
                embed_t= self.model.get_embedding(
                    dict((i, j.to(self.device)) for i, j in data['t'].items() if i in ['input_ids', 'attention_mask'])
                )
                embed_pos= self.model.get_embedding(
                    dict((i, j.to(self.device)) for i, j in data['pos'].items() if i in ['input_ids', 'attention_mask'])
                )
                embed_neg= self.model.get_embedding(
                    dict((i, j.to(self.device)) for i, j in data['neg'].items() if i in ['input_ids', 'attention_mask'])
                )
            return self.criterion(embed_t, embed_pos, embed_neg)
        
    def _train_on_epoch(self, index_grad, logging_step: Optional[int]= 100, step_save: Optional[int]= 1000): 
        self.model.train()
        total_loss, total_count= 0, 0
        step_loss, step_fr= 0, 0

        for idx, data in enumerate(self.dataloader_train): 
            with autocast():
                loss= self.compute_loss(data)
                loss /= self.grad_accum
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

            if (total_count + 1) % (logging_step * self.grad_accum) == 0: 
                print(f'Step: [{(total_count + 1 )/ (self.grad_accum):.0f}/{self.total_steps / self.epochs:.0f}], Loss: {(total_loss / total_count) * self.grad_accum}')
                step_loss = 0 
                step_fr = 0 
                index_grad[0] += 1 

            if (total_count + 1) % step_save ==0:
                torch.save({'step': idx + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                                },  self.ckpt_step)
        
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

    def fit(self, logging_step:Type[int]= 100, step_save: Type[int]= 1000, path_ckpt_epoch: Type[str]= 'best_ckpt.pt'):
        print(' START ')
        index_grad= [1] 
        log_loss = 1e9
        for epoch in range(1, self.epochs + 1): 
            print(f' EPOCH {epoch} ')
            train_loss= self._train_on_epoch(index_grad, logging_step, step_save)
            print(f'End of epoch {epoch} - loss: {train_loss}')            
            if len(self.path_dataeval) > 0:
                print(' EVALUATE ')
                val_loss= self._evaluate()
                print(f'Evaluate loss: {val_loss}')

                if val_loss < log_loss: 
                    log_loss = val_loss
                    print(f'Saving checkpoint have best {log_loss}')
                    torch.save({'epoch': epoch, 
                                'model_state_dict': self.model.state_dict(), 
                                'scheduler': self.scheduler.state_dict()}, 
                                path_ckpt_epoch)
                        


            if train_loss < log_loss: 
                log_loss = train_loss
                print(f'Saving checkpoint have best {log_loss}')
                torch.save({'epoch': epoch, 
                            'model_state_dict': self.model.state_dict(), 
                            'scheduler': self.scheduler.state_dict()},
                            path_ckpt_epoch)
    
        