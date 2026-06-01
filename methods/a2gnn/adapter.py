"""
A2GNN Adapter for PIDSMaker (AAAI 2024).

Core insight: Asymmetric propagation - source uses 0 propagation layers,
target uses 30 propagation layers. This tightens target risk bound.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

from ..base import BaseTransferMethod


class PropGCNConv(nn.Module):
    """Propagation-aware GCN convolution supporting variable propagation depth."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index, num_props=1):
        """Apply `num_props` rounds of message passing then linear transform."""
        # Normalize adjacency
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Propagate num_props times
        h = x
        for _ in range(num_props):
            h = torch.zeros_like(h).scatter_add_(0, col.unsqueeze(1).expand_as(h[row]), h[row] * norm.unsqueeze(1))
        
        return self.linear(h)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class A2GNNModel(nn.Module):
    """A2GNN: Asymmetric propagation for graph domain adaptation."""
    def __init__(self, in_dim, hid_dim=128, num_classes=2, num_layers=4,
                 s_pnums=0, t_pnums=10, dropout=0.5):
        super().__init__()
        self.s_pnums = s_pnums  # Source: no propagation (just linear)
        self.t_pnums = t_pnums  # Target: deep propagation
        self.dropout = dropout
        
        # Feature encoder (shared)
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GCNConv(in_dim, hid_dim))
        for _ in range(num_layers - 2):
            self.encoder_layers.append(GCNConv(hid_dim, hid_dim))
        self.encoder_layers.append(GCNConv(hid_dim, hid_dim))
        
        # Propagation layers (asymmetric)
        self.prop_layer = PropGCNConv(hid_dim, hid_dim)
        
        # Classifier
        self.classifier = nn.Linear(hid_dim, num_classes)
        
        # Domain discriminator
        self.domain_disc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hid_dim // 2, 2)
        )
    
    def encode(self, x, edge_index, is_source=True):
        """Encode with asymmetric propagation."""
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Asymmetric propagation
        pnums = self.s_pnums if is_source else self.t_pnums
        if pnums > 0:
            h = self.prop_layer(h, edge_index, num_props=pnums)
        
        return h
    
    def forward(self, x, edge_index, is_source=True, alpha=0.0):
        h = self.encode(x, edge_index, is_source)
        class_out = self.classifier(h)
        
        h_rev = GradReverse.apply(h, alpha)
        domain_out = self.domain_disc(h_rev)
        
        return class_out, domain_out, h


class A2GNNTransfer(BaseTransferMethod):
    """A2GNN asymmetric propagation DA for provenance graphs."""
    
    def __init__(self, cfg, device='cpu', epochs=200, lr=0.001, s_pnums=0, t_pnums=10):
        super().__init__(cfg, device)
        self.epochs = epochs
        self.lr = lr
        self.s_pnums = s_pnums
        self.t_pnums = t_pnums
    
    def train_source(self, source_graphs, source_labels=None):
        """Initialize model with source data."""
        sample = source_graphs[0]
        in_dim = sample.x.shape[1]
        self.model = A2GNNModel(
            in_dim=in_dim, s_pnums=self.s_pnums, t_pnums=self.t_pnums
        ).to(self.device)
        self.source_graphs = source_graphs
        return self.model
    
    def adapt_and_predict(self, target_graphs):
        """Joint adversarial training with asymmetric propagation."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        
        self.model.train()
        for epoch in range(self.epochs):
            p = epoch / self.epochs
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            
            for s_graph, t_graph in zip(self.source_graphs, target_graphs):
                s_graph = s_graph.to(self.device)
                t_graph = t_graph.to(self.device)
                optimizer.zero_grad()
                
                # Source (0 propagation)
                s_class, s_domain, _ = self.model(s_graph.x, s_graph.edge_index, is_source=True, alpha=alpha)
                # Target (deep propagation)
                t_class, t_domain, _ = self.model(t_graph.x, t_graph.edge_index, is_source=False, alpha=alpha)
                
                # Classification loss (source)
                if hasattr(s_graph, 'y') and s_graph.y is not None:
                    cls_loss = F.cross_entropy(s_class, s_graph.y.long())
                else:
                    cls_loss = F.cross_entropy(s_class, s_graph.x.argmax(dim=1))
                
                # Domain loss
                s_labels = torch.zeros(s_graph.num_nodes, dtype=torch.long, device=self.device)
                t_labels = torch.ones(t_graph.num_nodes, dtype=torch.long, device=self.device)
                domain_loss = F.cross_entropy(s_domain, s_labels) + F.cross_entropy(t_domain, t_labels)
                
                loss = cls_loss + 0.1 * domain_loss
                loss.backward()
                optimizer.step()
        
        # Predict on target
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for graph in target_graphs:
                graph = graph.to(self.device)
                logits, _, _ = self.model(graph.x, graph.edge_index, is_source=False)
                probs = F.softmax(logits, dim=1)
                scores = 1.0 - probs.max(dim=1).values
                all_scores.append(scores.cpu().numpy())
        
        scores = np.concatenate(all_scores)
        threshold = np.percentile(scores, 95)
        predictions = (scores > threshold).astype(int)
        
        return scores, predictions
