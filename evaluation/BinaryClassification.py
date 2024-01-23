import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import Any, List

class BinaryClassificationEvaluator:
    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        labels: List[int],
        name: str = "",
        batch_size: int = 32,
        write_csv: bool = True,
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
    
        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size

        self.csv_file = "binary_classification_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "cossim_accuracy",
            "cossim_accuracy_threshold",
            "cossim_f1",
            "cossim_precision",
            "cossim_recall",
            "cossim_f1_threshold",
            "cossim_ap",
            "manhattan_accuracy",
            "manhattan_accuracy_threshold",
            "manhattan_f1",
            "manhattan_precision",
            "manhattan_recall",
            "manhattan_f1_threshold",
            "manhattan_ap",
            "euclidean_accuracy",
            "euclidean_accuracy_threshold",
            "euclidean_f1",
            "euclidean_precision",
            "euclidean_recall",
            "euclidean_f1_threshold",
            "euclidean_ap",
            "dot_accuracy",
            "dot_accuracy_threshold",
            "dot_f1",
            "dot_precision",
            "dot_recall",
            "dot_f1_threshold",
            "dot_ap",
        ]

    def compute_metrices(self, model, tokenizer):
        toke_sentences1 = [tokenizer._tokenizer(sent) for sent in self.sentences1]
        toke_sentences2 = [tokenizer._tokenizer(sent) for sent in self.sentences2]

        embeddings1 = [model.get_embedding(dict( (i, j.to(self.device)) for i,j in sent.items())) for sent in toke_sentences1]
        embeddings2 = [model.get_embedding(dict( (i, j.to(self.device)) for i,j in sent.items())) for sent in toke_sentences2]

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
    
    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold
    
    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold
    
    def __call__(self, model, tokenizer, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"


        scores = self.compute_metrices(model, tokenizer)

        # Main score is the max of Average Precision (AP)
        main_score = max(scores[short_name]["ap"] for short_name in scores)

        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if "_" in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return main_score


        