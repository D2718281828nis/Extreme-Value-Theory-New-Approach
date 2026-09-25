"""Component 2: GAT source localization with event-level cross-validation."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GraphSample:
    features: np.ndarray
    edge_index: np.ndarray
    source_node: int


@dataclass(frozen=True)
class CrossValidationResult:
    fold_top1_accuracy: np.ndarray
    fold_mean_reciprocal_rank: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray


def _torch_modules():
    import torch
    from torch_geometric.nn import GATv2Conv

    return torch, GATv2Conv


class _GATFactory:
    @staticmethod
    def build(n_features: int, hidden: int):
        torch, GATv2Conv = _torch_modules()

        class SourceGAT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gat1 = GATv2Conv(
                    n_features, hidden, heads=2, concat=True, dropout=0.1
                )
                self.gat2 = GATv2Conv(2 * hidden, 1, heads=1, concat=False, dropout=0.0)

            def forward(self, x, edges):
                return self.gat2(torch.relu(self.gat1(x, edges)), edges).squeeze(-1)

        return SourceGAT()


def cross_validate_gat(
    samples: list[GraphSample],
    n_splits: int = 3,
    epochs: int = 40,
    hidden: int = 8,
    seed: int = 42,
) -> CrossValidationResult:
    """Train/test splits are made by independent event realizations, never by nodes."""
    if len(samples) < n_splits * 2 or n_splits < 2:
        raise ValueError("Need at least two samples per fold")
    from sklearn.model_selection import KFold

    torch, _ = _torch_modules()
    rng = np.random.default_rng(seed)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = np.zeros((len(samples), len(samples[0].features)), dtype=float)
    top1, mrr = [], []
    for fold, (train_ids, test_ids) in enumerate(folds.split(samples)):
        train_features = np.concatenate([samples[i].features for i in train_ids])
        mean, std = train_features.mean(0), np.maximum(train_features.std(0), 1e-6)
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        model = _GATFactory.build(samples[0].features.shape[1], hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            losses = []
            for i in rng.permutation(train_ids):
                sample = samples[int(i)]
                x = torch.tensor((sample.features - mean) / std, dtype=torch.float32)
                edges = torch.tensor(sample.edge_index, dtype=torch.long)
                logits = model(x, edges)
                losses.append(
                    torch.nn.functional.cross_entropy(
                        logits[None, :], torch.tensor([sample.source_node])
                    )
                )
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
        fold_ranks = []
        model.eval()
        with torch.no_grad():
            for i in test_ids:
                sample = samples[int(i)]
                x = torch.tensor((sample.features - mean) / std, dtype=torch.float32)
                edges = torch.tensor(sample.edge_index, dtype=torch.long)
                prob = torch.softmax(model(x, edges), dim=0).numpy()
                probabilities[i] = prob
                order = np.argsort(-prob)
                fold_ranks.append(int(np.where(order == sample.source_node)[0][0]) + 1)
        top1.append(np.mean(np.asarray(fold_ranks) == 1))
        mrr.append(np.mean(1 / np.asarray(fold_ranks)))
    return CrossValidationResult(
        np.asarray(top1),
        np.asarray(mrr),
        probabilities,
        np.asarray([s.source_node for s in samples]),
    )
