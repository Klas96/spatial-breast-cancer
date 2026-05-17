"""
Graph neural network for spatial domain prediction.
Nodes: Visium spots (features = gene expression PCs)
Edges: spatial neighbours
Task: predict tissue domain labels (tumour / stroma / immune / normal)

Input:  results/05_nhood_enrichment.h5ad
Output: results/06_gnn_metrics.txt, results/06_gnn_umap.png
"""

import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

RESULTS_DIR = Path("results")
EPOCHS = 300
LR = 1e-3
HIDDEN_DIM = 64


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)


def build_graph(adata: sc.AnnData) -> tuple[Data, LabelEncoder]:
    # Node features: PCA embedding
    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float)

    # Edge index from spatial neighbour graph
    adj = adata.obsp["spatial_connectivities"].tocoo()
    edge_index = torch.tensor(
        np.vstack([adj.row, adj.col]), dtype=torch.long
    )

    # Labels: dominant cell type as proxy for tissue domain
    le = LabelEncoder()
    labels = le.fit_transform(adata.obs["dominant_cell_type"].values)
    y = torch.tensor(labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y), le


def split_masks(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    def to_mask(idx):
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m

    return to_mask(train_idx), to_mask(val_idx), to_mask(test_idx)


def train(model, data, train_mask, val_mask, optimizer):
    train_losses, val_accs = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = out[val_mask].argmax(dim=1)
            acc = (pred == data.y[val_mask]).float().mean().item()
        train_losses.append(loss.item())
        val_accs.append(acc)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {acc:.3f}")

    return train_losses, val_accs


def plot_training(losses, accs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].plot(accs)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "06_gnn_training.png", dpi=150)
    plt.close()


def evaluate(model, data, test_mask, le, adata) -> None:
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).numpy()

    report = classification_report(
        data.y[test_mask].numpy(),
        pred[test_mask],
        target_names=le.classes_,
    )
    print(report)
    (RESULTS_DIR / "06_gnn_metrics.txt").write_text(report)

    adata.obs["gnn_pred"] = le.inverse_transform(pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sq.pl.spatial_scatter(adata, color="dominant_cell_type", ax=axes[0], img=False)
    axes[0].set_title("Ground truth (dominant cell type)")
    sq.pl.spatial_scatter(adata, color="gnn_pred", ax=axes[1], img=False)
    axes[1].set_title("GNN prediction")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "06_gnn_spatial_pred.png", dpi=150)
    plt.close()
    print("Saved: results/06_gnn_spatial_pred.png")


def main():
    adata = sc.read(RESULTS_DIR / "05_nhood_enrichment.h5ad")
    sq.gr.spatial_neighbors(adata, coord_type="visium")

    data, le = build_graph(adata)
    train_mask, val_mask, test_mask = split_masks(data.num_nodes)

    n_classes = len(le.classes_)
    model = GCN(data.num_node_features, HIDDEN_DIM, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    losses, accs = train(model, data, train_mask, val_mask, optimizer)
    plot_training(losses, accs)
    evaluate(model, data, test_mask, le, adata)


if __name__ == "__main__":
    main()
