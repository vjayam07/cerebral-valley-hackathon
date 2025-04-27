import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.utils import to_undirected

from typing import Tuple, Dict, List
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from new_train import RFEncoder, RFPredictor, RSSIDataset
from new_train import pool_mask, load_walkable_nodes, make_adjacent_edges, make_ray_edges

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## loading model .pt weights
def load_gnn(fp='best_model.pt'):
    state_dict = torch.load(fp, 
                            map_location=torch.device(device),
                            weights_only=True
                        )
    hidden_dim = 64
    model = RFPredictor(hidden_dim)
    model.load_state_dict(state_dict)
    return model

# -----------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------

def eval_epoch(model, data, loader, device):
    model.eval()
    total = 0
    with torch.no_grad():
        for tx, rx, y in tqdm(loader):
            tx, rx, y = tx.to(device), rx.to(device), y.to(device)
            total += F.mse_loss(model(data, tx, rx), y, reduction="sum").item()
    return total / len(loader.dataset)


from collections import deque

def snap_tx_to_walkable(tx_i, tx_j, id_map, cell_size=4, search_radius=50):
    """
    Return (node_id, coarse_row, coarse_col, manhattan_dist_cells)
    Always succeeds if *any* walkable cell exists within search_radius.
    """
    # 1. start from the coarse cell that contains the transmitter
    start_r = tx_i // cell_size
    start_c = tx_j // cell_size
    if (start_r, start_c) in id_map:
        nid = id_map[(start_r, start_c)]
        return nid, start_r, start_c, 0

    # 2. BFS spiral outwards until we hit the first walkable cell
    Q = deque([(start_r, start_c, 0)])
    seen = { (start_r, start_c) }
    while Q:
        r, c, d = Q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in seen or d+1 > search_radius:
                continue
            if (nr, nc) in id_map:                # <-- FOUND !
                return id_map[(nr,nc)], nr, nc, d+1
            seen.add((nr,nc))
            Q.append((nr,nc, d+1))

    raise RuntimeError("No walkable cell found within search_radius")

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main(args):
    #* 1) dataset/graph construction
    coords, id_map, (Wc, Hc) = load_walkable_nodes(args.mask_path, args.cell_size)

    df = pd.read_csv(args.csv_path, delimiter=args.delim)
    df2 = df[df['transmitter'] == 'tx2']

    row0 = df2.iloc[0]
    print(row0['tx_location_i'], row0["tx_location_j"])
    # 3. is that receiver *and* transmitter inside the mask?
    if (row0.tx_location_i // args.cell_size, row0.tx_location_j // args.cell_size) not in id_map:
        node_id, i_snap, j_snap, _ = snap_tx_to_walkable(row0['tx_location_i'], row0['tx_location_j'], id_map)
        df['tx_location_i'] = id_map[node_id]
        df['tx_location_j'] = id_map[node_id]
        print(i_snap, j_snap)
    
    print((row0.tx_location_i // args.cell_size, row0.tx_location_j // args.cell_size) in id_map)
    
    tx2 = RSSIDataset(df2, id_map, args.cell_size)
    print(len(tx2))
    test_dl = DataLoader(tx2, batch_size=args.batch_size)
    print(len(test_dl))


    #* 2) Initializing the edges of the graph
    adj = make_adjacent_edges(coords, id_map)
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
    data = data.to(device)

    #* 3) model
    model = load_gnn()
    model = model.to(device)
    test_score = eval_epoch(model, data, test_dl, device)

    print(f"MSE accuracy on tx2 dataset: {test_score}")

# -----------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("RF propagation GNN ‑ coarse grid")
    p.add_argument("--mask_path", type=Path, default="walkable_mask.png")
    p.add_argument("--csv_path", type=Path, required=True)
    p.add_argument("--delim", default=",", help="CSV delimiter")
    p.add_argument("--cell_size", type=int, default=4, help="pixels per super‑cell")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--ckpt_path", type=Path, default="best_model.pt")
    main(p.parse_args())



# def main():
#     df = pd.read_csv('train_data/training_walks.csv')
#     df_tx2 = df[df['transmitter'] == 'tx2']

#     model = load_gnn()

#     cell_size = 4

#     coords, id_map, (Wc, Hc) = load_walkable_nodes('train_data/walkable_mask.png', cell_size)
#     adj = make_adjacent_edges(coords, id_map)

#     CACHE = f"train_data/ray_edges_cs{cell_size}.pt"
#     blob = torch.load(CACHE)
#     assert blob["cell_size"] == cell_size
#     ray = blob["edge_index"]

#     data = HeteroData()
#     # node features: normalised coarse‑grid coords (x=j, y=i)
#     xy = coords[:, [1, 0]].astype(np.float32)
#     xy[:, 0] /= Wc; xy[:, 1] /= Hc
#     data["pixel"].x = torch.from_numpy(xy)
#     def to_idx(e):
#         return to_undirected(torch.tensor(e, dtype=torch.long).t())
#     data["pixel", "adjacent", "pixel"].edge_index = to_idx(adj)
#     data["pixel", "ray", "pixel"].edge_index = to_idx(ray)

#     df_tx2['predicted_RSSI'] = np.nan
#     df_tx2['MSE_loss'] = np.nan

#     ts = np.arange(0, len(df_tx2), 500)
#     for t in ts:
#         dp = df_tx2.iloc[t]
        
#         dp['predicted_RSSI'] = model(data, t, t)


