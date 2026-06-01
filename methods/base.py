"""Base class for all cross-domain transfer methods."""
import os
import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseTransferMethod(ABC):
    """Base interface for provenance graph domain adaptation methods."""
    
    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device
        self.model = None
    
    @abstractmethod
    def train_source(self, source_graphs, source_labels=None):
        """Train on source domain data."""
        pass
    
    @abstractmethod
    def adapt_and_predict(self, target_graphs):
        """Adapt to target domain and generate predictions.
        
        Returns:
            scores: np.array of anomaly scores per node
            predictions: np.array of binary predictions per node
        """
        pass
    
    def evaluate(self, predictions, scores, ground_truth_nids, total_nodes):
        """Compute standard metrics: TP, TN, FP, FN, MCC, F1, ADP."""
        y_true = np.zeros(total_nodes)
        for nid in ground_truth_nids:
            if nid < total_nodes:
                y_true[nid] = 1
        
        y_pred = predictions
        
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        
        # MCC
        denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = ((tp*tn) - (fp*fn)) / denom if denom > 0 else 0.0
        
        # F1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        # ADP (simplified): fraction of attacks detected at various thresholds
        adp = rec  # simplified as recall for now
        
        return {
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'MCC': round(mcc, 5), 'F1': round(f1, 5), 'ADP': round(adp, 3)
        }
