import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, List

class BinaryClassification:
    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        labels: List[int]
    ):
        self.distance= lambda x, y: 1- torch.cosine_similarity(x, y)
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
    

    def compute_metrices(self, model, margin: float = 0.5):
        TP, TN, FP, FN = 0, 0, 0, 0
        for sent1, sent2, label in tqdm(zip(self.sentences1, self.sentences2, self.labels)):
            embeddings1 = model.encode([sent1])
            embeddings2 = model.encode([sent2])

            cosine_distance = 1 - self.distance(embeddings1, embeddings2)

            if cosine_distance >= margin and label == 0:
                TN += 1
            if cosine_distance >= margin and label == 1:
                FN += 1
            if cosine_distance < margin and label == 0:
                FP += 1
            if cosine_distance < margin and label == 1:
                TP += 1

        accuracy = (TP + TN) / len(self.sentences1)
        recall  = TP / (TP + TN)
        precision = TP / (TP + FP)
        f1 = (2 * recall * precision) / (recall + precision)
        return {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1": f1 
        }
    


        