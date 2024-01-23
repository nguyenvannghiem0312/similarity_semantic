import torch 
from torch import nn 
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 5): 
        super(TripletLoss, self).__init__()
        self.margin= margin
        self.distance= lambda x, y: 1- torch.cosine_similarity(x, y)

    def forward(self, embedding_text, embedding_pos, embedding_neg): 
        
        distance_pos = self.distance(embedding_text, embedding_pos)
        distance_neg = self.distance(embedding_text, embedding_neg)

        losses = F.relu(distance_pos - distance_neg + self.margin)
        return losses.mean()
    