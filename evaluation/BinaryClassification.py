import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import Any, List

class BinaryClassificationEvaluator:
    def __init__(
        self,
        data: List[dict],
        name: str = "",
        batch_size: int = 32,
        write_csv: bool = True,
    ):
        self.data = data
        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        self.csv_file = "binary_classification_evaluation" + ("_" + name if name else "") + "_results.csv"

        self.csv_headers = [
            "epoch",
            "steps",
            "cossim_accuracy",
            "cossim_accuracy_threshold",
        ]

    def compute_metrices(self, model):
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(
            sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]

        labels = np.asarray(self.labels)
        output_scores = {}
        for short_name, name, scores, reverse in [
            ["cossim", "Cosine-Similarity", cosine_scores, True],
            ["manhattan", "Manhattan-Distance", manhattan_distances, False],
            ["euclidean", "Euclidean-Distance", euclidean_distances, False],
            ["dot", "Dot-Product", dot_scores, True],
        ]:
            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores * (1 if reverse else -1))

            logger.info(
                "Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold)
            )
            logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
            logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

            output_scores[short_name] = {
                "accuracy": acc,
                "accuracy_threshold": acc_threshold,
                "f1": f1,
                "f1_threshold": f1_threshold,
                "precision": precision,
                "recall": recall,
                "ap": ap,
            }

        return output_scores
    
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"


        