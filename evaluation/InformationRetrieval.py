import torch
from torch import Tensor
import logging
from tqdm import trange
import os
import numpy as np
from typing import List, Dict, Set, Callable
import heapq

class InformationRetrieval():
    def __init__(
        self,
        queries: Dict[str, str],  # qid => query
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
    ):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.mrr_at_k = map_at_k
    