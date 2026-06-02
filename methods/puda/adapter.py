"""
PUDA Adapter for PIDSMaker cross-domain transfer.

PUDA (Provenance-based Unsupervised Domain Adaptation) uses:
1. Word2Vec features -> Linear projection
2. GCN/GAT structural encoder (2 layers + BatchNorm)
3. Three parallel heads:
   - Task Classifier: predicts unified node type (PROCESS, FILE, NETWORK, OTHER)
   - Salience Scorer: residual MLP -> sigmoid anomaly score
   - Domain Discriminator: GRL + MLP for adversarial domain alignment

Training: Joint source task + salience loss + adversarial domain loss (GRL)
Inference: Salience scores as anomaly indicators on target domain

Reference: https://github.com/anyuan1999/Puda
"""
import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch_geometric.nn import GCNConv, GATv2Conv, BatchNorm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ..base import BaseTransferMethod


# ============ Gradient Reversal Layer ============

class GradientReversalFunction(Function):
    """GRL: forward=identity, backward=negate*lambda."""
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_param, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_param=1.0):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_param)

    def set_lambda(self, val):
        self.lambda_param = val


# ============ Unified Node Type Mapping ============

class UnifiedType:
    PROCESS = 0
    FILE = 1
    NETWORK = 2
    OTHER = 3
    COUNT = 4


class UnifiedMap:
    """Maps dataset-specific node types to 4 unified categories."""
    TRACE_MAP = {
        0: UnifiedType.PROCESS, 1: UnifiedType.OTHER, 2: UnifiedType.FILE,
        3: UnifiedType.FILE, 4: UnifiedType.FILE, 5: UnifiedType.PROCESS,
        6: UnifiedType.FILE, 7: UnifiedType.NETWORK, 8: UnifiedType.OTHER,
        9: UnifiedType.FILE, 10: UnifiedType.NETWORK, 11: UnifiedType.FILE
    }
    OPTC_MAP = {
        0: UnifiedType.PROCESS, 1: UnifiedType.NETWORK,
        2: UnifiedType.FILE, 3: UnifiedType.FILE
    }
    CADETS_MAP = {
        0: UnifiedType.PROCESS, 1: UnifiedType.FILE, 2: UnifiedType.NETWORK,
        3: UnifiedType.FILE, 4: UnifiedType.NETWORK, 5: UnifiedType.FILE
    }
    THEIA_MAP = {
        0: UnifiedType.PROCESS, 1: UnifiedType.OTHER, 2: UnifiedType.FILE,
        3: UnifiedType.NETWORK, 4: UnifiedType.NETWORK, 5: UnifiedType.OTHER
    }
    FIVEDIRECTIONS_MAP = {
        0: UnifiedType.PROCESS, 1: UnifiedType.FILE, 2: UnifiedType.OTHER,
        3: UnifiedType.FILE, 4: UnifiedType.NETWORK, 5: UnifiedType.FILE,
        6: UnifiedType.NETWORK, 7: UnifiedType.PROCESS, 8: UnifiedType.PROCESS
    }

    @staticmethod
    def get_map(dataset_name):
        name = dataset_name.lower()
        if 'trace' in name:
            return UnifiedMap.TRACE_MAP
        if 'optc' in name:
            return UnifiedMap.OPTC_MAP
        if 'cadets' in name:
            return UnifiedMap.CADETS_MAP
        if 'theia' in name:
            return UnifiedMap.THEIA_MAP
        if 'five' in name:
            return UnifiedMap.FIVEDIRECTIONS_MAP
        # Default: treat as OTHER for all types
        return {}

    @staticmethod
    def convert_node_types(raw_labels, dataset_name, device):
        """Convert raw dataset node type labels to unified types."""
        mapping = UnifiedMap.get_map(dataset_name)
        if not mapping:
            return torch.full_like(raw_labels, UnifiedType.OTHER)
        max_id = max(mapping.keys()) + 1
        lookup = torch.zeros(max_id, dtype=torch.long, device=device)
        for k, v in mapping.items():
            lookup[k] = v
        # Clamp to avoid OOB
        clamped = raw_labels.clamp(0, max_id - 1)
        return lookup[clamped]


# ============ Focal Loss ============

class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-7
        pred = pred.float().clamp(eps, 1 - eps)
        target = target.float()
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


# ============ PUDA Model ============

class PUDAModel(nn.Module):
    """
    Cross-domain PUDA model with 3 parallel heads.

    Architecture:
    - Word2Vec projection -> GCN/GAT encoder -> BatchNorm
    - Head 1: Task Classifier (unified node type, 4 classes)
    - Head 2: Salience Scorer (residual MLP -> sigmoid)
    - Head 3: Domain Discriminator (GRL -> MLP -> 2 classes)
    """
    def __init__(self, word2vec_dim=30, gat_hidden=64, num_node_types=4,
                 num_domains=2, heads=4, num_layers=2, dropout=0.3, use_gcn=True):
        super().__init__()
        self.gat_hidden = gat_hidden
        self.dropout = dropout
        self.use_gcn = use_gcn

        # Word2Vec feature projection
        self.word2vec_proj = nn.Linear(word2vec_dim, gat_hidden)

        # Structural encoder (GCN or GAT)
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = gat_hidden
            if use_gcn:
                self.conv_layers.append(GCNConv(in_dim, gat_hidden))
            else:
                self.conv_layers.append(
                    GATv2Conv(in_dim, gat_hidden, heads=heads, dropout=dropout, concat=False)
                )
            self.batch_norms.append(BatchNorm(gat_hidden))

        # Feature normalization
        self.feature_norm = nn.BatchNorm1d(gat_hidden)

        # Head 1: Task Classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(gat_hidden, gat_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden // 2, num_node_types)
        )

        # Head 2: Salience Scorer (with residual)
        self.salience_fc1 = nn.Linear(gat_hidden, gat_hidden)
        self.salience_bn1 = nn.BatchNorm1d(gat_hidden)
        self.salience_fc2 = nn.Linear(gat_hidden, gat_hidden // 2)
        self.salience_bn2 = nn.BatchNorm1d(gat_hidden // 2)
        self.salience_fc3 = nn.Linear(gat_hidden // 2, 1)
        self.salience_dropout = nn.Dropout(dropout)

        # Head 3: Domain Discriminator (with GRL)
        self.gradient_reversal = GradientReversalLayer(lambda_param=1.0)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(gat_hidden, gat_hidden),
            nn.BatchNorm1d(gat_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden, gat_hidden // 2),
            nn.BatchNorm1d(gat_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden // 2, num_domains)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, word2vec_features, edge_index, return_features=False):
        """
        Forward pass.

        Args:
            word2vec_features: [N, word2vec_dim]
            edge_index: [2, E]
            return_features: whether to also return the encoded features

        Returns:
            task_logits: [N, 4]
            salience_scores: [N, 1] in [0, 1]
            domain_logits: [N, 2]
            features (optional): [N, gat_hidden]
        """
        # 1. Project word2vec features
        x = F.relu(self.word2vec_proj(word2vec_features))

        # 2. GCN/GAT encoding
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Feature normalization
        x = self.feature_norm(x)
        features = x

        # 3. Parallel heads
        # Task classifier
        task_logits = self.task_classifier(features)

        # Salience scorer with residual
        sal_h1 = self.salience_fc1(features)
        sal_h1 = self.salience_bn1(sal_h1)
        sal_h1 = F.relu(sal_h1)
        sal_h1 = self.salience_dropout(sal_h1)
        sal_h1 = sal_h1 + features  # residual connection

        sal_h2 = self.salience_fc2(sal_h1)
        sal_h2 = self.salience_bn2(sal_h2)
        sal_h2 = F.relu(sal_h2)
        sal_h2 = self.salience_dropout(sal_h2)

        salience_scores = torch.sigmoid(self.salience_fc3(sal_h2))

        # Domain discriminator (with GRL)
        reversed_features = self.gradient_reversal(features)
        domain_logits = self.domain_discriminator(reversed_features)

        if return_features:
            return task_logits, salience_scores, domain_logits, features
        return task_logits, salience_scores, domain_logits

    def set_grl_lambda(self, val):
        self.gradient_reversal.set_lambda(val)

    def predict_salience(self, word2vec_features, edge_index):
        """Inference: return salience scores only."""
        self.eval()
        with torch.no_grad():
            _, salience_scores, _ = self.forward(word2vec_features, edge_index)
        return salience_scores.squeeze()


# ============ GRL Lambda Schedule ============

def compute_grl_lambda(epoch, total_epochs, max_lambda=1.0):
    """Progressive GRL schedule: 2*max/(1+exp(-10*p)) - max"""
    p = epoch / total_epochs
    lam = 2 * max_lambda / (1 + np.exp(-10 * p)) - max_lambda
    return float(lam)


# ============ PUDA Transfer Adapter ============

class PUDATransfer(BaseTransferMethod):
    """
    PUDA domain adaptation for provenance graph transfer.

    Trains on source domain with:
    - Task loss (unified node type classification)
    - Salience loss (Focal Loss on attack labels)
    - Adversarial domain loss (GRL for domain-invariant features)

    Inference on target domain using salience scores as anomaly indicators.
    """

    def __init__(self, cfg, device='cpu', epochs=30, lr=0.005, batch_size=2048,
                 gat_hidden=64, num_layers=2, dropout=0.3, use_gcn=True,
                 lambda_sal=3.0, lambda_adv=1.0, max_grl_lambda=1.0,
                 word2vec_dim=30, threshold_strategy='percentile_top1'):
        super().__init__(cfg, device)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.gat_hidden = gat_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gcn = use_gcn
        self.lambda_sal = lambda_sal
        self.lambda_adv = lambda_adv
        self.max_grl_lambda = max_grl_lambda
        self.word2vec_dim = word2vec_dim
        self.threshold_strategy = threshold_strategy

        # These are set during training
        self._source_name = None
        self._target_name = None
        self._source_graph = None
        self._target_graph = None

    def _detect_feature_dim(self, graphs):
        """Detect feature dimension from graph data."""
        for g in graphs:
            if hasattr(g, 'x') and g.x is not None:
                return g.x.shape[1]
        return self.word2vec_dim

    def _build_pyg_graph(self, graphs, dataset_name=None):
        """
        Merge multiple graph snapshots into a single PyG Data object.
        Handles both PIDSMaker temporal batches and standard PyG Data objects.
        """
        all_x = []
        all_edge_index = []
        all_y = []
        node_offset = 0

        for g in graphs:
            if hasattr(g, 'x') and g.x is not None:
                x = g.x
            else:
                # Fall back to one-hot or zero features
                num_nodes = g.num_nodes if hasattr(g, 'num_nodes') else g.edge_index.max().item() + 1
                x = torch.zeros(num_nodes, self.word2vec_dim)

            all_x.append(x)

            if hasattr(g, 'edge_index') and g.edge_index is not None:
                ei = g.edge_index + node_offset
                all_edge_index.append(ei)

            if hasattr(g, 'y') and g.y is not None:
                all_y.append(g.y)
            else:
                all_y.append(torch.zeros(x.shape[0], dtype=torch.long))

            node_offset += x.shape[0]

        merged_x = torch.cat(all_x, dim=0)
        merged_ei = torch.cat(all_edge_index, dim=1) if all_edge_index else torch.zeros(2, 0, dtype=torch.long)
        merged_y = torch.cat(all_y, dim=0)

        graph = Data(
            x=merged_x,
            y=merged_y,
            edge_index=merged_ei,
        )
        graph.n_id = torch.arange(graph.num_nodes)
        return graph

    def _compute_unified_types(self, graph, dataset_name):
        """Infer unified node types from graph features or raw labels."""
        if hasattr(graph, 'node_type') and graph.node_type is not None:
            return UnifiedMap.convert_node_types(graph.node_type, dataset_name, graph.x.device)

        # If features are one-hot encoded (e.g., PIDSMaker's only_type), argmax to get raw type
        if graph.x is not None and graph.x.shape[1] <= 15:
            raw_types = graph.x.argmax(dim=1)
            return UnifiedMap.convert_node_types(raw_types, dataset_name, graph.x.device)

        # Default: assign OTHER to all nodes
        return torch.full((graph.num_nodes,), UnifiedType.OTHER, dtype=torch.long, device=graph.x.device)

    def train_source(self, source_graphs, source_labels=None, target_graphs=None,
                     source_name='source', target_name='target'):
        """
        Train PUDA model with source domain supervision + adversarial domain alignment.

        Args:
            source_graphs: List of source PyG graphs
            source_labels: Optional explicit labels (overrides graph.y)
            target_graphs: List of target PyG graphs (required for adversarial training)
            source_name: Dataset name for unified type mapping
            target_name: Dataset name for unified type mapping
        """
        self._source_name = source_name
        self._target_name = target_name

        # Build merged graphs
        source_graph = self._build_pyg_graph(source_graphs, source_name).to(self.device)
        self._source_graph = source_graph

        if target_graphs is not None:
            target_graph = self._build_pyg_graph(target_graphs, target_name).to(self.device)
            self._target_graph = target_graph
        else:
            target_graph = None

        # Detect feature dimension
        feat_dim = source_graph.x.shape[1]

        # Build model
        self.model = PUDAModel(
            word2vec_dim=feat_dim,
            gat_hidden=self.gat_hidden,
            num_node_types=UnifiedType.COUNT,
            num_domains=2,
            heads=4,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_gcn=self.use_gcn
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )

        # Compute unified node types for source
        source_unified_types = self._compute_unified_types(source_graph, source_name)

        # Task loss: class-balanced CrossEntropy
        type_counts = torch.bincount(source_unified_types, minlength=UnifiedType.COUNT).float()
        type_weights = torch.where(type_counts > 0, 1.0 / type_counts, torch.zeros_like(type_counts))
        type_weights = type_weights / type_weights.sum() * UnifiedType.COUNT
        criterion_task = nn.CrossEntropyLoss(weight=type_weights.to(self.device))

        # Salience loss: FocalLoss for attack/benign imbalance
        focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

        # Setup NeighborLoader for source
        source_mask = torch.ones(source_graph.num_nodes, dtype=torch.bool, device=self.device)
        source_loader = NeighborLoader(
            source_graph, num_neighbors=[10, 5],
            batch_size=self.batch_size, input_nodes=source_mask, shuffle=True
        )

        # Setup target loader if available
        target_loader = None
        target_iter = None
        if target_graph is not None:
            target_mask = torch.ones(target_graph.num_nodes, dtype=torch.bool, device=self.device)
            target_loader = NeighborLoader(
                target_graph, num_neighbors=[10, 5],
                batch_size=self.batch_size, input_nodes=target_mask, shuffle=True
            )
            target_iter = iter(target_loader)

        # Training loop
        self.model.train()
        best_loss = float('inf')

        print(f"  [PUDA] Training {self.epochs} epochs | feat_dim={feat_dim} | "
              f"hidden={self.gat_hidden} | {'GCN' if self.use_gcn else 'GAT'}")
        print(f"  [PUDA] Source: {source_graph.num_nodes} nodes, {source_graph.num_edges} edges")
        if target_graph:
            print(f"  [PUDA] Target: {target_graph.num_nodes} nodes, {target_graph.num_edges} edges")

        for epoch in range(self.epochs):
            # Progressive GRL lambda
            grl_lambda = compute_grl_lambda(epoch, self.epochs, self.max_grl_lambda)
            self.model.set_grl_lambda(grl_lambda)

            epoch_loss_task = 0
            epoch_loss_sal = 0
            epoch_loss_adv = 0
            num_batches = 0

            for source_batch in source_loader:
                source_batch = source_batch.to(self.device)
                optimizer.zero_grad()

                # Source forward
                s_task, s_sal, s_dom = self.model(
                    source_graph.x[source_batch.n_id], source_batch.edge_index
                )

                # Task loss (on source unified types)
                batch_types = source_unified_types[source_batch.n_id[:source_batch.batch_size]]
                loss_task = criterion_task(s_task[:source_batch.batch_size], batch_types)

                # Salience loss (on source attack labels)
                batch_labels = source_graph.y[source_batch.n_id[:source_batch.batch_size]].float().unsqueeze(1)
                loss_sal = focal_loss(s_sal[:source_batch.batch_size], batch_labels)

                # NaN guard
                if torch.isnan(loss_sal):
                    loss_sal = F.binary_cross_entropy(
                        s_sal[:source_batch.batch_size].clamp(1e-7, 1-1e-7), batch_labels
                    )

                # Domain adversarial loss
                loss_adv = torch.tensor(0.0, device=self.device)
                if target_loader is not None:
                    try:
                        target_batch = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_loader)
                        target_batch = next(target_iter)

                    target_batch = target_batch.to(self.device)
                    _, t_sal, t_dom = self.model(
                        target_graph.x[target_batch.n_id], target_batch.edge_index
                    )

                    # Domain labels: source=0, target=1
                    s_dom_labels = torch.zeros(s_dom.size(0), dtype=torch.long, device=self.device)
                    t_dom_labels = torch.ones(t_dom.size(0), dtype=torch.long, device=self.device)

                    loss_dom_s = F.cross_entropy(s_dom, s_dom_labels)
                    loss_dom_t = F.cross_entropy(t_dom, t_dom_labels)
                    loss_adv = (loss_dom_s + loss_dom_t) / 2

                # Total loss
                loss_total = loss_task + self.lambda_sal * loss_sal + self.lambda_adv * loss_adv

                if not torch.isnan(loss_total):
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                epoch_loss_task += loss_task.item()
                epoch_loss_sal += loss_sal.item()
                epoch_loss_adv += loss_adv.item()
                num_batches += 1

            if num_batches > 0:
                avg_total = epoch_loss_task/num_batches + epoch_loss_sal/num_batches + epoch_loss_adv/num_batches
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1}/{self.epochs} | Task={epoch_loss_task/num_batches:.4f} "
                          f"Sal={epoch_loss_sal/num_batches:.4f} Adv={epoch_loss_adv/num_batches:.4f} "
                          f"GRL_lambda={grl_lambda:.3f}")

            # Memory cleanup
            if (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return self.model

    def adapt_and_predict(self, target_graphs, target_name=None):
        """
        Run inference on target domain using salience scores.

        Args:
            target_graphs: List of target PyG graphs
            target_name: Dataset name (for logging)

        Returns:
            scores: np.array of salience scores per node (0-1, higher = more anomalous)
            predictions: np.array of binary predictions (1 = anomalous)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_source() first.")

        self.model.eval()

        # Build target graph
        if self._target_graph is not None and target_graphs is None:
            target_graph = self._target_graph
        else:
            target_graph = self._build_pyg_graph(
                target_graphs, target_name or self._target_name
            ).to(self.device)

        # Batched inference
        eval_mask = torch.ones(target_graph.num_nodes, dtype=torch.bool, device=self.device)
        eval_loader = NeighborLoader(
            target_graph, num_neighbors=[15, 10],
            batch_size=2000, input_nodes=eval_mask, shuffle=False
        )

        all_salience = []
        all_indices = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(self.device)
                _, salience, _ = self.model(
                    target_graph.x[batch.n_id], batch.edge_index
                )
                batch_size = batch.batch_size if hasattr(batch, 'batch_size') else len(batch.n_id)
                all_salience.append(salience[:batch_size].cpu())
                all_indices.append(batch.n_id[:batch_size].cpu())

        # Merge and deduplicate
        all_salience = torch.cat(all_salience, dim=0).squeeze().numpy()
        all_indices = torch.cat(all_indices, dim=0).numpy()

        # Build full score array (deduplicated)
        node_scores = {}
        for idx, sal in zip(all_indices, all_salience):
            idx = int(idx)
            if idx not in node_scores:
                node_scores[idx] = float(sal)

        scores = np.zeros(target_graph.num_nodes)
        for nid, sal in node_scores.items():
            if nid < len(scores):
                scores[nid] = sal

        # Apply threshold strategy
        predictions = self._apply_threshold(scores)

        print(f"  [PUDA] Target inference: {target_graph.num_nodes} nodes, "
              f"{predictions.sum()} predicted anomalous ({100*predictions.mean():.2f}%)")

        return scores, predictions

    def _apply_threshold(self, scores):
        """Apply threshold strategy to convert salience scores to binary predictions."""
        strategy = self.threshold_strategy

        if strategy.startswith('fixed_'):
            thresh = float(strategy.replace('fixed_', ''))
            return (scores > thresh).astype(int)
        elif strategy.startswith('percentile_top'):
            pct_str = strategy.replace('percentile_top', '').replace('%', '')
            top_pct = float(pct_str)
            percentile = 100 - top_pct
            thresh = np.percentile(scores, percentile)
            return (scores > thresh).astype(int)
        else:
            # Default: top 1% as anomalous
            thresh = np.percentile(scores, 99)
            return (scores > thresh).astype(int)

    def save_model(self, save_dir):
        """Save trained model weights."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'puda_model.pth'))
        print(f"  [PUDA] Model saved to {save_dir}")

    def load_model(self, model_path, feat_dim=None):
        """Load model from checkpoint."""
        state_dict = torch.load(model_path, map_location=self.device)

        # Infer feature dim from checkpoint
        if feat_dim is None:
            feat_dim = state_dict['word2vec_proj.weight'].shape[1]

        # Detect GCN vs GAT
        use_gcn = any('conv_layers' in k and '.lin.weight' in k for k in state_dict.keys())

        self.model = PUDAModel(
            word2vec_dim=feat_dim,
            gat_hidden=self.gat_hidden,
            num_node_types=UnifiedType.COUNT,
            num_domains=2,
            heads=4,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_gcn=use_gcn
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        print(f"  [PUDA] Model loaded from {model_path} (feat_dim={feat_dim}, {'GCN' if use_gcn else 'GAT'})")
        return self.model
