"""
Training script for RF signal propagation modeling using a Graph Neural Network.

* Grid nodes are taken from a binary walkable‐mask image – every white pixel is treated as a walkable node.
* Two edge types are created:
    1. **adjacent** – 4‑connected neighbours on the grid.
    2. **ray** – Bresenham line between every TX/RX pair in the dataset (unique edges collected globally).
* A heterogeneous graph is built with the two edge types using **torch_geometric**.
* A simple two‑layer HeteroConv encoder creates node embeddings that are fed to an MLP regressor to predict RSSI for each (TX, RX) pair.
* Training/validation losses are logged to **Weights & Biases (wandb)** – the code assumes you are already logged‑in (`wandb login`).

Author: ChatGPT – April 2025
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import wandb

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.utils import to_undirected

# ---------------
# Helper functions
# ---------------

def load_walkable_nodes(img_path: Path) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """Return coordinates array (N, 2) and dict mapping (i, j) → node_id for all walkable pixels.
    Walkable = white pixels (value > 0).
    """
    img = Image.open(img_path).convert("L")
    mask = np.array(img) > 0  # bool array
    coords = np.argwhere(mask)  # (N, 2) – (row=i, col=j)
    # NOTE: np.argwhere returns (row, col) already.
    id_map: Dict[Tuple[int, int], int] = {tuple(coord): idx for idx, coord in enumerate(coords)}
    return coords, id_map


def make_adjacent_edges(coords: np.ndarray, id_map: Dict[Tuple[int, int], int]) -> List[Tuple[int, int]]:
    """4‑neighbour undirected edges across walkable nodes."""
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    edges = []
    for coord in coords:
        for d in directions:
            nb = tuple(coord + d)
            if nb in id_map:
                edges.append((id_map[tuple(coord)], id_map[nb]))
    return edges


def bresenham(p0: Tuple[int, int], p1: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Integer Bresenham line between p0 and p1 (both inclusive)."""
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy  # error value e_xy
    line = []
    while True:
        line.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return line


def make_ray_edges(df: pd.DataFrame, id_map: Dict[Tuple[int, int], int]) -> List[Tuple[int, int]]:
    """Collect *unique* undirected edges that lie on Bresenham rays between all TX/RX pairs."""
    edges_set = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trace rays"):
        tx = (int(row["tx_location_i"]), int(row["tx_location_j"]))
        rx = (int(row["i"]), int(row["j"]))
        if tx not in id_map or rx not in id_map:
            # skip pairs that lie outside the walkable mask
            continue
        path = bresenham(tx, rx)
        for p0, p1 in zip(path[:-1], path[1:]):
            if p0 in id_map and p1 in id_map:
                u, v = id_map[p0], id_map[p1]
                edges_set.add((u, v))
                edges_set.add((v, u))  # undirected
    return list(edges_set)


# ---------------
# PyG dataset for link‑prediction/regression
# ---------------

class RSSIDataset(Dataset):
    """Each item returns (tx_id, rx_id, rssi)."""

    def __init__(self, df: pd.DataFrame, id_map: Dict[Tuple[int, int], int]):
        self.samples = []
        for _, row in df.iterrows():
            tx = (int(row["tx_location_i"]), int(row["tx_location_j"]))
            rx = (int(row["i"]), int(row["j"]))
            if tx in id_map and rx in id_map:
                self.samples.append(
                    (
                        id_map[tx],
                        id_map[rx],
                        float(row["rssi"]),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------
# Model
# ---------------

class RFEncoder(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        # Two GATv2 layers per edge type → aggregated by mean
        self.conv1 = HeteroConv(
            {
                "adjacent": GATv2Conv(in_dim, hidden_dim, heads=2, concat=False),
                "ray": GATv2Conv(in_dim, hidden_dim, heads=2, concat=False),
            },
            aggr="mean",
        )
        self.conv2 = HeteroConv(
            {
                "adjacent": GATv2Conv(hidden_dim, out_dim, heads=2, concat=False),
                "ray": GATv2Conv(hidden_dim, out_dim, heads=2, concat=False),
            },
            aggr="mean",
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class RFPredictor(nn.Module):
    def __init__(self, node_dim: int = 64):
        super().__init__()
        self.enc = RFEncoder(out_dim=node_dim)
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )

    def forward(self, data: HeteroData, tx_ids: torch.Tensor, rx_ids: torch.Tensor):
        # Encode full graph once → embeddings
        x_dict = self.enc(data.x_dict, data.edge_index_dict)
        embeds = x_dict["pixel"]  # (N, D)
        h_tx = embeds[tx_ids]
        h_rx = embeds[rx_ids]
        out = self.mlp(torch.cat([h_tx, h_rx], dim=-1)).squeeze(-1)
        return out


# ---------------
# Training utils
# ---------------

def train_epoch(model, data, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for tx_ids, rx_ids, rssis in loader:
        tx_ids = tx_ids.to(device)
        rx_ids = rx_ids.to(device)
        rssis = rssis.to(device)
        optimizer.zero_grad()
        pred = model(data, tx_ids, rx_ids)
        loss = F.mse_loss(pred, rssis)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * tx_ids.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, data, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for tx_ids, rx_ids, rssis in loader:
            tx_ids = tx_ids.to(device)
            rx_ids = rx_ids.to(device)
            rssis = rssis.to(device)
            pred = model(data, tx_ids, rx_ids)
            loss = F.mse_loss(pred, rssis)
            total_loss += loss.item() * tx_ids.size(0)
    return total_loss / len(loader.dataset)


# ---------------
# Main routine
# ---------------

def main(args):
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # 1. Load graph nodes & edges
    coords, id_map = load_walkable_nodes(args.mask_path)
    print(f"Walkable nodes: {coords.shape[0]:,}")

    adj_edges = make_adjacent_edges(coords, id_map)
    print(f"Adjacent edges: {len(adj_edges):,}")



    df = pd.read_csv(args.csv_path, delimiter=args.delim)
    ray_edges = make_ray_edges(df, id_map)
    print(f"Ray edges: {len(ray_edges):,}")

    # 2. Build HeteroData graph (undirected)
    data = HeteroData()
    # Node features: normalised coordinates in [0,1]
    h, w = Image.open(args.mask_path).size  # PIL gives (width, height)
    x = coords[:, [1, 0]].astype(np.float32)  # (j, i)
    x[:, 0] /= w
    x[:, 1] /= h
    data["pixel"].x = torch.from_numpy(x)

    def to_edge_index(edge_list):
        tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return to_undirected(tensor)

    data["pixel", "adjacent", "pixel"].edge_index = to_edge_index(adj_edges)
    data["pixel", "ray", "pixel"].edge_index = to_edge_index(ray_edges)

    data = data.to(args.device)

    # 3. Build dataset of pairs
    full_ds = RSSIDataset(df, id_map)
    val_len = int(len(full_ds) * args.val_ratio)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

    # 4. Model, optimizer
    model = RFPredictor(node_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, data, train_loader, optimizer, args.device)
        val_loss = eval_epoch(model, data, val_loader, args.device)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.ckpt_path)
            wandb.run.summary["best_val_loss"] = best_val

    print("Training complete. Best val loss:", best_val)


# ---------------
# CLI
# ---------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF propagation GNN trainer")
    parser.add_argument("--mask_path", type=Path, default="walkable_mask.png", help="Path to walkable mask image")
    parser.add_argument("--csv_path", type=Path, required=True, help="Path to CSV dataset")
    parser.add_argument("--delim", type=str, default="\t", help="CSV delimiter (default tab)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", type=Path, default="ckpts/best_model.pt")
    parser.add_argument("--wandb_project", type=str, default="rf_propagation_gnn")
    parser.add_argument("--run_name", type=str, default="exp1")
    args = parser.parse_args()
    main(args)
