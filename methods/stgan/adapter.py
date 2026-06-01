"""
STGAN Adapter for PIDSMaker cross-domain transfer.

STGAN (WWW 2025) uses GAT + TGN + Multi-Head Self-Attention for node classification.
It does NOT have a native DA mechanism - we use direct zero-shot transfer.

Integration: Train on source provenance graph, apply directly to target.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ..base import BaseTransferMethod


class GATEncoder(nn.Module):
    """3-layer GAT encoder (from STGAN)."""
    def __init__(self, in_dim, hid_dim=64, out_dim=30, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hid_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hid_dim * heads, hid_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hid_dim * heads, out_dim, heads=1, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hid_dim * heads)
        self.bn2 = nn.BatchNorm1d(hid_dim * heads)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class STGANModel(nn.Module):
    """Simplified STGAN model for transfer experiments."""
    def __init__(self, in_dim, hid_dim=64, gat_out=30, num_classes=2):
        super().__init__()
        self.gat = GATEncoder(in_dim, hid_dim, gat_out)
        self.classifier = nn.Sequential(
            nn.Linear(gat_out, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hid_dim, num_classes)
        )
    
    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        out = self.classifier(h)
        return out, h  # logits, embeddings


class STGANTransfer(BaseTransferMethod):
    """STGAN zero-shot transfer (train on source, predict on target directly)."""
    
    def __init__(self, cfg, device='cpu', epochs=50, lr=0.001):
        super().__init__(cfg, device)
        self.epochs = epochs
        self.lr = lr
    
    def train_source(self, source_graphs, source_labels=None):
        """Train STGAN on source domain."""
        # Get feature dimension from first graph
        sample = source_graphs[0]
        in_dim = sample.x.shape[1] if hasattr(sample, 'x') and sample.x is not None else 16
        
        self.model = STGANModel(in_dim=in_dim, num_classes=2).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for graph in source_graphs:
                graph = graph.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(graph.x, graph.edge_index)
                
                # Use node labels if available, else self-supervised
                if hasattr(graph, 'y') and graph.y is not None:
                    loss = F.cross_entropy(logits, graph.y.long())
                else:
                    # Self-supervised: reconstruct node type from embeddings
                    loss = F.cross_entropy(logits, graph.x.argmax(dim=1))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return self.model
    
    def adapt_and_predict(self, target_graphs):
        """Direct inference on target (no adaptation)."""
        self.model.eval()
        all_scores = []
        
        with torch.no_grad():
            for graph in target_graphs:
                graph = graph.to(self.device)
                logits, _ = self.model(graph.x, graph.edge_index)
                # Anomaly score = loss per node (higher = more anomalous)
                probs = F.softmax(logits, dim=1)
                scores = 1.0 - probs.max(dim=1).values  # uncertainty as anomaly score
                all_scores.append(scores.cpu().numpy())
        
        scores = np.concatenate(all_scores)
        threshold = np.percentile(scores, 95)  # top 5% as anomalous
        predictions = (scores > threshold).astype(int)
        
        return scores, predictions
