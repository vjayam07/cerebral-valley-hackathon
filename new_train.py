"""
Training script for RF signal‑propagation modelling on a **coarsened grid** with a Graph Neural Network.

Changes vs. the previous version
--------------------------------
* **`--cell_size S`** (≥1) pools the binary walkable mask into non‑overlapping `S×S` super‑cells.
  * A super‑cell is *walkable* if **any** pixel inside it is walkable.
  * All TX/RX coordinates from the CSV are integer‑divided by `S` before graph lookup so training data instantly matches the coarse grid.
* The rest of the pipeline (adjacent‑edge construction, ray tracing with Bresenham, GATv2‑based encoder, training loop, wandb logging) stays identical.
---
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import wandb

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.utils import to_undirected

# -----------------------------------------------------------
# Utility: down‑sample a boolean mask by max‑pooling S×S blocks
# -----------------------------------------------------------

# def pool_mask(mask: np.ndarray, S: int) -> np.ndarray:
#     """Return pooled mask with shape (H//S, W//S), where *any* walkable pixel
#     inside a block marks the entire super‑cell as walkable."""
#     if S == 1:
#         return mask.astype(bool)
#     H, W = mask.shape
#     H_trim = (H // S) * S
#     W_trim = (W // S) * W
#     mask = mask[:H_trim, :W_trim]
#     # reshape & max‑pool
#     pooled = mask.reshape(H_trim // S, S, W_trim // S, S).max(axis=(1, 3))
#     return pooled.astype(bool)

def pool_mask(mask: np.ndarray, cell_size: int = 4):
    h, w = mask.shape
    h_crop = (h // cell_size) * cell_size         # largest multiple ≤ h
    w_crop = (w // cell_size) * cell_size
    mask = mask[:h_crop, :w_crop]                 # throw away the ragged fringe

    # now safe to reshape/pool
    pooled = mask.reshape(h_crop // cell_size, cell_size,
                          w_crop // cell_size, cell_size
                         ).max(axis=(1, 3))       # OR .mean(...)
    return pooled


def downsample_coord(row: int, col: int, S: int) -> Tuple[int, int]:
    """Map original‑resolution (row, col) to coarse grid indices."""
    return row // S, col // S


# -----------------------------------------------------------
# Load walkable nodes & build id_map
# -----------------------------------------------------------

def load_walkable_nodes(mask_path: Path, cell_size: int) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    img = Image.open(mask_path).convert("L")  # grayscale
    mask = np.array(img) > 0  # bool
    pooled = pool_mask(mask, cell_size)
    coords = np.argwhere(pooled)  # (row, col)
    id_map = {tuple(coord): idx for idx, coord in enumerate(coords)}
    return coords, id_map, pooled.shape[::-1]  # coords, map, (Wc, Hc)


# -----------------------------------------------------------
# Graph edge construction helpers
# -----------------------------------------------------------

def make_adjacent_edges(coords: np.ndarray, id_map: Dict[Tuple[int, int], int]) -> List[Tuple[int, int]]:
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)
    edges = []
    for coord in coords:
        for d in dirs:
            nb = tuple(coord + d)
            if nb in id_map:
                edges.append((id_map[tuple(coord)], id_map[nb]))
    return edges


def bresenham(p0: Tuple[int, int], p1: Tuple[int, int]) -> List[Tuple[int, int]]:
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
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


def make_ray_edges(df: pd.DataFrame, id_map: Dict[Tuple[int, int], int], S: int) -> List[Tuple[int, int]]:
    edges = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trace rays"):
        tx = downsample_coord(int(row.tx_location_i), int(row.tx_location_j), S)
        rx = downsample_coord(int(row.i), int(row.j), S)
        if tx not in id_map or rx not in id_map:
            continue
        for p0, p1 in zip(bresenham(tx, rx)[:-1], bresenham(tx, rx)[1:]):
            if p0 in id_map and p1 in id_map:
                u, v = id_map[p0], id_map[p1]
                edges.add((u, v))
                edges.add((v, u))
    return list(edges)


# -----------------------------------------------------------
# PyTorch‑Geometric dataset of (TX,RX,rssi)
# -----------------------------------------------------------

class RSSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_map: Dict[Tuple[int, int], int], S: int):
        self.samples = []
        for _, row in df.iterrows():
            tx = downsample_coord(int(row.tx_location_i), int(row.tx_location_j), S)
            rx = downsample_coord(int(row.i), int(row.j), S)
            if tx in id_map and rx in id_map:
                self.samples.append((id_map[tx], id_map[rx], float(row.rssi)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# -----------------------------------------------------------
# GNN model: two GATv2 layers per edge type + MLP head
# -----------------------------------------------------------

class RFEncoder(nn.Module):
    def __init__(self, in_dim: int = 2, hid: int = 64, heads: int = 2):
        super().__init__()
        self.conv1 = HeteroConv(
            {
                ('pixel', 'adjacent', 'pixel'):  # ← 3-tuple!
                    GATv2Conv(in_dim, hid,
                              heads=heads,
                              concat=False,
                              add_self_loops=False),
                ('pixel', 'ray', 'pixel'):       # ← 3-tuple!
                    GATv2Conv(in_dim, hid,
                              heads=heads,
                              concat=False,
                              add_self_loops=False),
            },
            aggr='mean',
        )

        self.conv2 = HeteroConv(
            {
                ('pixel', 'adjacent', 'pixel'):
                    GATv2Conv(hid, hid,
                              heads=heads,
                              concat=False,
                              add_self_loops=False),
                ('pixel', 'ray', 'pixel'):
                    GATv2Conv(hid, hid,
                              heads=heads,
                              concat=False,
                              add_self_loops=False),
            },
            aggr='mean',
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


# class RFEncoder(nn.Module):
#     def __init__(self, in_dim=2, hid=64, out_dim=64):
#         super().__init__()
#         self.conv1 = HeteroConv({
#             "adjacent": GATv2Conv(in_dim, hid, heads=2, concat=False, add_self_loops=False),
#             "ray":      GATv2Conv(in_dim, hid, heads=2, concat=False, add_self_loops=False),
#         }, aggr="mean")
#         self.conv2 = HeteroConv({
#             "adjacent": GATv2Conv(hid, out_dim, heads=2, concat=False, add_self_loops=False),
#             "ray":      GATv2Conv(hid, out_dim, heads=2, concat=False, add_self_loops=False),
#         }, aggr="mean")

#     def forward(self, x_dict, edge_index_dict):
#         x = self.conv1(x_dict, edge_index_dict)
#         x = {k: F.relu(v) for k, v in x.items()}
#         x = self.conv2(x, edge_index_dict)
#         return x

class RFPredictor(nn.Module):
    def __init__(self, node_dim=64):
        super().__init__()
        self.enc = RFEncoder() # ! changed from out_dim=node_dim
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )

    def forward(self,
                data: HeteroData,
                tx_idx,
                rx_idx):
        # --- 1. get pixel embeddings ------------------------------------
        z = self.enc(data.x_dict, data.edge_index_dict)["pixel"]

        # --- 2. normalise the indices -----------------------------------
        tx_idx = torch.as_tensor(tx_idx, dtype=torch.long, device=z.device)
        rx_idx = torch.as_tensor(rx_idx, dtype=torch.long, device=z.device)

        if tx_idx.dim() == 0:
            tx_idx = tx_idx.unsqueeze(0)
            rx_idx = rx_idx.unsqueeze(0)

        # --- 3. gather & predict ----------------------------------------
        pair_emb = torch.cat([z[tx_idx], z[rx_idx]], dim=-1)
        pred = self.mlp(pair_emb).squeeze(-1)
        return pred

    # def forward(self, data: HeteroData, tx, rx):
    #     z = self.enc(data.x_dict, data.edge_index_dict)["pixel"]
    #     return self.mlp(torch.cat([z[tx], z[rx]], dim=-1)).squeeze(-1)

# -----------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------

def train_epoch(model, data, loader, opt, device):
    model.train()
    total = 0
    for tx, rx, y in tqdm(loader):
        tx, rx, y = tx.to(device), rx.to(device), y.to(device).float()
        opt.zero_grad()

        pred = model(data, tx, rx)
        loss = F.mse_loss(pred, y)
        
        loss.backward()
        opt.step()

        total += loss.item() * len(y)
    return total / len(loader.dataset)

def eval_epoch(model, data, loader, device):
    model.eval(); total = 0
    with torch.no_grad():
        for tx, rx, y in loader:
            tx, rx, y = tx.to(device), rx.to(device), y.to(device)
            total += F.mse_loss(model(data, tx, rx), y, reduction="sum").item()
    return total / len(loader.dataset)

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main(args):
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # 1) graph construction
    coords, id_map, (Wc, Hc) = load_walkable_nodes(args.mask_path, args.cell_size)
    adj = make_adjacent_edges(coords, id_map)
    df = pd.read_csv(args.csv_path, delimiter=args.delim)

    # file_path = "ray_tracing.json"
    # if not os.path.exists(file_path):
    #     ray = make_ray_edges(df, id_map, args.cell_size)
    #     with open(file_path, 'w') as file:
    #         json.dump(ray, file)
    # else:
    #     with open(file_path, 'r') as file:
    #         ray = json.load(file)
    # * Saving ray tracing results
    CACHE = f"train_data/ray_edges_cs{args.cell_size}.pt"
    if not os.path.exists(CACHE):
        ray = make_ray_edges(df, id_map, args.cell_size)
        tmp = CACHE + ".tmp"
        torch.save({"cell_size": args.cell_size,
                    "edge_index": ray}, tmp)
        os.replace(tmp, CACHE)      # atomic move
    else:
        blob = torch.load(CACHE)
        assert blob["cell_size"] == args.cell_size
        ray = blob["edge_index"]
    

    data = HeteroData()
    # node features: normalised coarse‑grid coords (x=j, y=i)
    xy = coords[:, [1, 0]].astype(np.float32)
    xy[:, 0] /= Wc; xy[:, 1] /= Hc
    data["pixel"].x = torch.from_numpy(xy)
    def to_idx(e):
        return to_undirected(torch.tensor(e, dtype=torch.long).t())
    data["pixel", "adjacent", "pixel"].edge_index = to_idx(adj)
    data["pixel", "ray", "pixel"].edge_index = to_idx(ray)
    data = data.to(args.device)

    # 2) pair dataset
    full = RSSIDataset(df, id_map, args.cell_size)
    val_len = int(len(full) * args.val_ratio)
    train_ds, val_ds = random_split(full, [len(full)-val_len, val_len])
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size)

    # 3) model
    model = RFPredictor(args.hidden_dim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    for epoch in range(1, args.epochs+1):
        tr = train_epoch(model, data, train_ld, opt, args.device)
        vl = eval_epoch(model, data, val_ld, args.device)
        wandb.log({"epoch": epoch, "train_loss": tr, "val_loss": vl})
        print(f"Ep {epoch:03d}  train {tr:.4f}  val {vl:.4f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), args.ckpt_path)
            wandb.run.summary["best_val"] = best
    print("Done. Best val =", best)

# -----------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("RF propagation GNN ‑ coarse grid")
    p.add_argument("--mask_path", type=Path, default="walkable_mask.png")
    p.add_argument("--csv_path", type=Path, required=True)
    p.add_argument("--delim", default=",", help="CSV delimiter")
    p.add_argument("--cell_size", type=int, default=4, help="pixels per super‑cell")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_path", type=Path, default="best_model.pt")
    p.add_argument("--wandb_project", default="rf_propagation_gnn")
    p.add_argument("--run_name", default="coarse_grid_exp")
    main(p.parse_args())
