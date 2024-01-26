from typing import Any, List, Dict, Set, Callable

import torch
from torch import Tensor
from tqdm import tqdm
import numpy as np


class InformationRetrieval:
    def __init__(
        self,
        queries: Dict[str, str],  # qid => query {"index": "query"}
        corpus: Dict[str, str],  # cid => corpus {"index": "context"}
        relevant_docs: List[List[str]],  # [[qid, pid, nid]]
        mrr_at_k: int = 5,
        ndcg_at_k: int = 5,
        accuracy_at_k: int = 1,
        precision_recall_at_k: int = 1,
        map_at_k: int = 5,
    ):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.mrr_at_k = map_at_k

        self.emb_corpus = None
        

    def _embedding_corpus(self, model):
        contexts = [self.corpus[i] for i in self.corpus.keys()]
        contexts_emb = model.encode(contexts)
        return contexts_emb
    
    def _get_rank_k(self, model, query, contexts_emb, k):
        query_emb = model.encode([query] * len(contexts_emb)) 
        scores = model.calculate_cosine_similarity(query_emb, contexts_emb)
        topk_values, topk_indices = torch.topk(scores, k)
        return topk_indices

    def compute_metrices(self, model):
        contexts_emb = self._embedding_corpus(model)
        rank = []
        for query_id, pos_id, neg_id in tqdm(self.relevant_docs):
            query = self.queries.get(query_id, "")
            topk_indices = self._get_rank_k(model, query, contexts_emb, k=self.mrr_at_k)
            positions = torch.where(topk_indices == int(pos_id))[0].cpu().numpy()
            if len(positions) == 0:
                rank.append(-1)
            else:
                rank.append(positions[0] + 1)
            print(rank)
        return rank

