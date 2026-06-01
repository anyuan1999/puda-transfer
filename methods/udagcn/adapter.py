"""
UDAGCN Adapter for PIDSMaker (WWW 2020).

Core: Dual GCN (local + PPMI-global) + Attention fusion + Adversarial DA + Entropy minimization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, add_self_loops

from ..base import BaseTransferMethod


class PPMIConv(nn.Module):
    """PPMI-based GCN convolution (captures global structure via random walks)."""
    def __init__(self, in_dim, out_dim, path_len=10, num_walks=40):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.path_len = path_len
        self.num_walks = num_walks
        self._ppmi_cache = None
    
    def compute_ppmi(self, edge_index, num_nodes):
        """Compute PPMI matrix from random walks."""
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        # Normalize to transition matrix
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        trans = adj / deg
        
        # Random walk: compute co-occurrence
        walk = torch.eye(num_nodes, device=adj.device)
        cooccur = torch.zeros_like(adj)
        for _ in range(self.path_len):
            walk = walk @ trans
            cooccur += walk
        
        # PPMI
        cooccur = cooccur / self.path_len
        row_sum = cooccur.sum(dim=1, keepdim=True).clamp(min=1e-8)
        col_sum = cooccur.sum(dim=0, keepdim=True).clamp(min=1e-8)
        total = cooccur.sum().clamp(min=1e-8)
        
        pmi = torch.log((cooccur * total) / (row_sum * col_sum) + 1e-8)
        ppmi = torch.clamp(pmi, min=0)
        
        # Normalize PPMI
        ppmi_deg = ppmi.sum(dim=1, keepdim=True).clamp(min=1e-8)
        ppmi_norm = ppmi / ppmi_deg
        
        return ppmi_norm
    
    def forward(self, x, edge_index, num_nodes=None):
        if num_nodes is None:
            num_nodes = x.size(0)
        
        if self._ppmi_cache is None or self._ppmi_cache.size(0) != num_nodes:
            self._ppmi_cache = self.compute_ppmi(edge_index, num_nodes)
        
        # GCN with PPMI matrix
        out = self._ppmi_cache.to(x.device) @ x
        out = self.linear(out)
        return out


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class UDAGCNModel(nn.Module):
    """UDAGCN: Dual GCN + Attention + Domain Discriminator."""
    def __init__(self, in_dim, hid_dim=128, num_classes=2):
        super().__init__()
        # Local GCN (standard)
        self.local_gcn1 = GCNConv(in_dim, hid_dim)
        self.local_gcn2 = GCNConv(hid_dim, hid_dim)
        
        # Global GCN (PPMI-based)
        self.global_gcn1 = PPMIConv(in_dim, hid_dim)
        self.global_gcn2 = PPMIConv(hid_dim, hid_dim)
        
        # Attention fusion
        self.att_local = nn.Linear(hid_dim, 1)
        self.att_global = nn.Linear(hid_dim, 1)
        
        # Classifier
        self.classifier = nn.Linear(hid_dim, num_classes)
        
        # Domain discriminator
        self.domain_disc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, 2)
        )
    
    def encode(self, x, edge_index):
        """Dual encoder with attention fusion."""
        # Local path
        h_local = F.relu(self.local_gcn1(x, edge_index))
        h_local = self.local_gcn2(h_local, edge_index)
        
        # Global path (PPMI)
        h_global = F.relu(self.global_gcn1(x, edge_index))
        h_global = self.global_gcn2(h_global, edge_index)
        
        # Attention
        a_local = torch.sigmoid(self.att_local(h_local))
        a_global = torch.sigmoid(self.att_global(h_global))
        a_sum = a_local + a_global + 1e-8
        
        h = (a_local / a_sum) * h_local + (a_global / a_sum) * h_global
        return h
    
    def forward(self, x, edge_index, alpha=0.0):
        h = self.encode(x, edge_index)
        class_out = self.classifier(h)
        
        # Domain discrimination with gradient reversal
        h_rev = GradientReversal.apply(h, alpha)
        domain_out = self.domain_disc(h_rev)
        
        return class_out, domain_out, h


class UDAGCNTransfer(BaseTransferMethod):
    """UDAGCN domain adaptation for provenance graphs."""
    
    def __init__(self, cfg, device='cpu', epochs=200, lr=0.001):
        super().__init__(cfg, device)
        self.epochs = epochs
        self.lr = lr
    
    def train_source(self, source_graphs, source_labels=None):
        """Pre-train on source (called before adapt_and_predict)."""
        sample = source_graphs[0]
        in_dim = sample.x.shape[1]
        self.model = UDAGCNModel(in_dim=in_dim).to(self.device)
        self.source_graphs = source_graphs
        return self.model
    
    def adapt_and_predict(self, target_graphs):
        """Joint training with adversarial DA + entropy minimization."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        
        self.model.train()
        for epoch in range(self.epochs):
            # Gradually increase reversal strength
            p = epoch / self.epochs
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            alpha *= 0.05  # Scale down as in original paper
            
            total_loss = 0
            for s_graph, t_graph in zip(self.source_graphs, target_graphs):
                s_graph = s_graph.to(self.device)
                t_graph = t_graph.to(self.device)
                
                optimizer.zero_grad()
                
                # Source forward
                s_class, s_domain, s_h = self.model(s_graph.x, s_graph.edge_index, alpha)
                # Target forward
                t_class, t_domain, t_h = self.model(t_graph.x, t_graph.edge_index, alpha)
                
                # Source classification loss
                if hasattr(s_graph, 'y') and s_graph.y is not None:
                    cls_loss = F.cross_entropy(s_class, s_graph.y.long())
                else:
                    cls_loss = F.cross_entropy(s_class, s_graph.x.argmax(dim=1))
                
                # Domain adversarial loss
                s_domain_labels = torch.zeros(s_graph.num_nodes, dtype=torch.long, device=self.device)
                t_domain_labels = torch.ones(t_graph.num_nodes, dtype=torch.long, device=self.device)
                domain_loss = F.cross_entropy(s_domain, s_domain_labels) + \
                              F.cross_entropy(t_domain, t_domain_labels)
                
                # Target entropy minimization
                t_probs = F.softmax(t_class, dim=1)
                entropy_loss = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=1).mean()
                
                loss = cls_loss + domain_loss + 0.1 * entropy_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Predict on target
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for graph in target_graphs:
                graph = graph.to(self.device)
                logits, _, _ = self.model(graph.x, graph.edge_index)
                probs = F.softmax(logits, dim=1)
                scores = 1.0 - probs.max(dim=1).values
                all_scores.append(scores.cpu().numpy())
        
        scores = np.concatenate(all_scores)
        threshold = np.percentile(scores, 95)
        predictions = (scores > threshold).astype(int)
        
        return scores, predictions
