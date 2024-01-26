import os
import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_val = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        if self.best_val is None:
            self.best_val = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val = val_loss
            self.counter = 0
        return self.early_stop