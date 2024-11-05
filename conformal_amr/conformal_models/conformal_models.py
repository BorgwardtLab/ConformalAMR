import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import brentq, linprog
from scipy.special import expit, logit
from sklearn.linear_model import QuantileRegressor

# import torch


class ConformalPredictor:

    def __init__(self, config):
        self.n = config["cal_size"]
        self.alpha = config["alpha"]
        self.cal_clf_output = config["cal_clf_scores"]
        self.cal_labels = config["cal_labels"]
        self.cal_scores = self.nonconformity_score(self.cal_clf_output, self.cal_labels)
        self.qhat = None

    def nonconformity_score(self, classifier_scores, label=None):
        return 1 - classifier_scores[label == 1]

    def get_qhat(self):
        q_level = np.ceil((self.n + 1) * (1 - self.alpha)) / self.n
        return np.quantile(self.cal_scores, q_level, interpolation="higher")

    def predict(self, classifier_scores, labels=None):
        if self.qhat is None:
            self.qhat = self.get_qhat()

        return self.nonconformity_score(classifier_scores) <= self.qhat


def label_spreading_numpy(
    A, s, alpha=1.0, num_iters=1000, aggregation="max", softmax_factor=1, tol=1e-09
):
    """
    A: n x n np.ndarray
    s: d x n np.ndarray
    n: number of drugs
    d: number of calibrations
    aggregation: mean, max, softmax, selective_mean
    """
    A = A.astype(bool)
    np.fill_diagonal(A, 1)
    s = s.astype("float64").T
    n, d = s.shape
    if aggregation == "mean":
        norm_mask = A / A.sum(axis=1, keepdims=True)

    def pooling(mask, h):
        if aggregation == "max":
            h_neighbors = np.where(mask[:, :, None], h[None, :, :], -np.inf)
            h_agg = np.max(h_neighbors, axis=1)
        elif aggregation == "mean":
            h_agg = norm_mask @ h
        return h_agg

    constant_term = (1 - alpha) * s
    new_s = s

    if aggregation in ["max", "mean"]:
        for i in range(num_iters):
            s_prev = new_s
            new_s = pooling(A, new_s)
            new_s = alpha * new_s + constant_term
            if np.linalg.norm(new_s - s_prev) < tol:
                break
        return new_s.T
    elif aggregation in ["softmax", "filtered_mean"]:
        A = A[:, :, None]
        if aggregation == "softmax":
            temperature = softmax_factor * s.copy()
            temperature = temperature[:, None, :]
        for i in range(num_iters):
            s_prev = new_s
            new_s_expanded = new_s[None, :, :]
            if aggregation == "filtered_mean":
                A_mask = new_s_expanded >= new_s[:, None, :]
                A_mask = A & A_mask
                neighbor_counts = A_mask.sum(axis=1)
                new_s = np.where(A_mask, new_s_expanded, 0)
                new_s = np.sum(new_s, axis=1) / np.maximum(neighbor_counts, 1)
            else:
                scores = (new_s[:, None, :] * new_s_expanded) / temperature
                scores = np.where(A, scores, -np.inf)
                max_scores = scores.max(axis=1, keepdims=True)
                exp_scores = np.exp(scores - max_scores)
                exp_scores = np.where(A, exp_scores, 0)
                weights = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-10)
                new_s = np.sum(weights * new_s_expanded, axis=1)

            new_s = alpha * new_s + constant_term
            if np.linalg.norm(new_s - s_prev) < tol:
                break

        return new_s.T
    else:
        raise NotImplementedError


class KGConformalPredictor(ConformalPredictor):

    def __init__(self, config):
        self.drug_list = config["drug_list"]
        self.tau = config["tau"]
        self.cal_clf_output = config["cal_clf_scores"]
        self.agg_type = config["agg_type"]
        self.softmax_factor = 1
        if "softmax_factor" in config.keys():
            self.softmax_factor = config["softmax_factor"]
        self.create_graph()
        self.qhat_dict = None
        super().__init__(config)

    def create_graph(self):
        self.graph_edges = [
            ("Ceftriaxone", "Ceftazidime"),
            ("Ceftazidime", "Ceftriaxone"),
            ("Cefepime", "Ceftriaxone"),
            ("Cefepime", "Ceftazidime"),
            ("Imipenem", "Cefepime"),
            ("Imipenem", "Ceftriaxone"),
            ("Imipenem", "Ceftazidime"),
            ("Ertapenem", "Cefepime"),
            ("Ertapenem", "Ceftriaxone"),
            ("Ertapenem", "Ceftazidime"),
            ("Meropenem", "Cefepime"),
            ("Meropenem", "Ceftriaxone"),
            ("Meropenem", "Ceftazidime"),
            ("Meropenem", "Ertapenem"),
            ("Ciprofloxacin", "Levofloxacin"),
            ("Levofloxacin", "Ciprofloxacin"),
            ("Meropenem", "Amoxicillin-Clavulanic acid"),
            ("Ertapenem", "Amoxicillin-Clavulanic acid"),
            ("Imipenem", "Amoxicillin-Clavulanic acid"),
            ("Meropenem", "Piperacillin-Tazobactam"),
            ("Ertapenem", "Piperacillin-Tazobactam"),
            ("Imipenem", "Piperacillin-Tazobactam"),
            ("Piperacillin-Tazobactam", "Amoxicillin-Clavulanic acid"),
            ("Cefepime", "Piperacillin-Tazobactam"),
            ("Cefepime", "Amoxicillin-Clavulanic acid"),
            ("Ceftriaxone", "Amoxicillin-Clavulanic acid"),
            ("Ceftazidime", "Amoxicillin-Clavulanic acid"),
        ]

        G = nx.DiGraph()
        G.add_nodes_from([(i, {"drug": d}) for i, d in enumerate(self.drug_list)])
        index_dict = {drug: index for index, drug in enumerate(self.drug_list)}
        G.add_edges_from(
            [
                (index_dict[d1], index_dict[d2])
                for d1, d2 in self.graph_edges
                if d1 in self.drug_list and d2 in self.drug_list
            ]
        )
        self.Graph = G
        self.graph_adjecency_matrix = nx.adjacency_matrix(
            self.Graph
        ).toarray() + np.eye(self.drug_list.shape[0])

    def nonconformity_score(self, classifier_scores, label=None):

        classifier_scores = label_spreading_numpy(
            self.graph_adjecency_matrix,
            classifier_scores,
            alpha=self.tau,
            softmax_factor=self.softmax_factor,
            aggregation=self.agg_type,
        )
        return 1 - classifier_scores

    def get_qhat(self):
        self.cal_scores[~(self.cal_labels == 1)] = -np.inf
        max_label_scores = np.max(self.cal_scores, axis=1)
        q_level = np.ceil((self.n + 1) * (1 - self.alpha)) / self.n
        return np.quantile(max_label_scores, q_level, interpolation="higher")

    def predict(self, classifier_scores, labels=None):
        if self.qhat is None:
            self.qhat = self.get_qhat()

        return self.nonconformity_score(classifier_scores) <= self.qhat

    def get_qhat_dict(self, alpha_list):

        qhat_dict = {}
        self.cal_scores[~(self.cal_labels == 1)] = -np.inf
        max_label_scores = np.max(self.cal_scores, axis=1)
        for alpha in alpha_list:
            q_level = np.ceil((self.n + 1) * (1 - alpha)) / self.n
            qhat = np.quantile(max_label_scores, q_level, interpolation="higher")
            qhat_dict[alpha] = qhat

        return qhat_dict

    def predict_all_alpha(self, classifier_scores, alpha_list=[], labels=None):
        if self.qhat_dict is None:
            self.qhat_dict = self.get_qhat_dict(alpha_list)

        return {
            alpha: (self.nonconformity_score(classifier_scores) <= qhat)
            for alpha, qhat in self.qhat_dict.items()
        }
