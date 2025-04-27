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

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main(args):
    # 1) graph construction
    coords, id_map, (Wc, Hc) = load_walkable_nodes(args.mask_path, args.cell_size)
    adj = make_adjacent_edges(coords, id_map)
    df = pd.read_csv(args.csv_path, delimiter=args.delim)

    df_no2 = df[df['transmitter'] != 'tx2'] # removing one transmitter for testing purposes

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
    data = data.to(device)

    df2 = df[df['transmitter'] == 'tx2']
    tx2 = RSSIDataset(df2, id_map, args.cell_size)
    print(len(tx2))
    test_dl = DataLoader(tx2, batch_size=args.batch_size)
    print(len(test_dl))

    # 3) model
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


