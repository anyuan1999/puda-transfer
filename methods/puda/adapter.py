"""
PUDA Adapter - Wraps the existing zero-shot transfer mechanism.

This is the baseline: train on source, apply directly to target with weight resizing.
Already implemented in transfer_inference.py.
"""
from ..base import BaseTransferMethod


class PUDATransfer(BaseTransferMethod):
    """PUDA: Zero-shot provenance-based transfer (already implemented)."""
    
    def __init__(self, cfg, device='cpu'):
        super().__init__(cfg, device)
    
    def train_source(self, source_graphs, source_labels=None):
        """Uses PIDSMaker's native training (via main.py)."""
        raise NotImplementedError(
            "PUDA uses PIDSMaker's native pipeline. "
            "Run: python pidsmaker/main.py <detector> <source_dataset>"
        )
    
    def adapt_and_predict(self, target_graphs):
        """Uses transfer_inference.py for zero-shot transfer."""
        raise NotImplementedError(
            "PUDA uses transfer_inference.py. "
            "Run: python transfer_inference.py <detector> <source> <target>"
        )
