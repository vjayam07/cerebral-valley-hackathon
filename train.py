"""
Baseline implementation of the GNN‑based radio‑propagation model from
"Data‑Driven Radio Propagation Modeling using Graph Neural Networks"
(Adrien Bufort et al., 2025).

The script covers four essential steps:
  1.  Converting raw RSSI point‑measurements around a transmitter into a
      *single* grid graph that contains
         • local 4‑neighbour spatial edges, and
         • ray‑tracing edges that link each pixel to the transmitter along
           the direct path.
  2.  A `torch_geometric`‑compatible `Dataset` that lazily yields a
      `Data` object per transmitter zone.  Only the pixels where a
      measurement exists have a supervision target – this is conveyed by
      the boolean `mask` attribute so that the loss can be computed with
      *partial* labels.
  3.  A message‑passing neural network that follows the architecture in
      the paper: node/edge encoders → FiLM conditioning on scalar antenna
      features → *N* stacked Meta‑layers → node decoder.
  4.  A minimal training / validation loop that minimises the masked MSE.

The code is intentionally lightweight and self‑contained: complex
features such as building masks, multi‑resolution tiling, or advanced ray
intersections can be added on top of this baseline without changing the
core model.

Tested with:
  * PyTorch ≥ 2.2
  * PyTorch Geometric ≥ 2.5
  * pandas, numpy, torch‑scatter

Author: (your name here)
"""
from __future__ import annotations

import math
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean

###############################################################################
# Graph construction helpers                                                  #
###############################################################################

def build_grid_edges(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return 4‑neighbour *bidirectional* grid edges and their attributes.

    Edge attributes = (Δx, Δy, edge_type) where *edge_type==0* denotes a
    spatial (grid) connection.
    """
    edges: List[Tuple[int, int]] = []
    attrs: List[Tuple[float, float, float]] = []
    for r in range(height):
        for c in range(width):
            nid = r * width + c
            # south
            if r + 1 < height:
                nid2 = (r + 1) * width + c
                edges.extend([(nid, nid2), (nid2, nid)])
                attrs.extend([(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)])
            # east
            if c + 1 < width:
                nid2 = r * width + c + 1
                edges.extend([(nid, nid2), (nid2, nid)])
                attrs.extend([(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def build_ray_edges(
    height: int,
    width: int,
    cell_size: float,
    tx_xy: Tuple[float, float],
    edge_type_value: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one *direct* edge from the transmitter to every pixel (and back).

    The transmitter is assumed to lie at coordinates (0, 0) in the local
    map reference frame *before* any pixel‑to‑metre conversion – place the
    grid such that the centre pixel coincides with the transmitter for
    the simplest match with the paper.  If a different placement is
    needed, translate the coordinates before calling this helper.

    Edge attributes = (radius, θ [rad], edge_type) with *edge_type==1*
    identifying ray‑tracing edges.
    """
    tx_x, tx_y = tx_xy
    xs = np.arange(width) * cell_size + cell_size / 2 - width * cell_size / 2
    ys = np.arange(height) * cell_size + cell_size / 2 - height * cell_size / 2
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten() - tx_x
    yv = yv.flatten() - tx_y

    radii = np.hypot(xv, yv)
    thetas = np.arctan2(yv, xv)
    num_nodes = height * width
    tx_id = num_nodes  # we append a dedicated *transmitter* node at the end

    senders = np.full(num_nodes, tx_id, dtype=np.int64)
    receivers = np.arange(num_nodes, dtype=np.int64)
    # add reverse edges so that information can flow back to the GNN
    edge_index = torch.tensor(
        np.vstack(
            [np.hstack([senders, receivers]), np.hstack([receivers, senders])]
        ),
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        np.vstack(
            [
                np.hstack([radii, radii]),
                np.hstack([thetas, -thetas]),  # opposite angle on return
                np.full(2 * num_nodes, edge_type_value),
            ]
        ).T,
        dtype=torch.float32,
    )
    return edge_index, edge_attr, tx_id

###############################################################################
# Dataset                                                                     #
###############################################################################

class RSSIGraphDataset(Dataset):
    """Lazy dataset that yields one *graph* per transmitter area.

    The CSV files are expected to contain *at least* the following columns:
        tx_x, tx_y … transmitter longitude/latitude (or projected metres)
        x,   y     … receiver coordinate in the same system as above
        rssi       … received signal power in dB

    All points from *one* CSV belong to the same cell/sector.  You can
    split your measurement archive by eNodeB ID or by hexagonal site –
    whichever makes sense for your application.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        grid_size: int = 128,
        cell_size: float = 5.0,
        context_km: float = 0.5,
    ) -> None:
        super().__init__(root)
        self.files = sorted(Path(root).glob("*.csv"))
        if not self.files:
            raise RuntimeError(f"No CSV files found in {root!s}.")
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.half_extent = grid_size * cell_size / 2
        self.context_km = context_km

    # ---------------------------------------------------------------------
    def len(self) -> int:  # type: ignore[override]
        return len(self.files)

    # ---------------------------------------------------------------------
    def get(self, idx: int) -> Data:  # type: ignore[override]
        df = pd.read_csv(self.files[idx])
        tx_x, tx_y = df.loc[0, ["tx_x", "tx_y"]].values.astype(float)
        scalar_feats = torch.as_tensor(
            df.loc[0, ["frequency_mhz", "antenna_height_m", "eirp_dbm"]].values,
            dtype=torch.float32,
        )

        # ------------------------------------------------------------------
        # 1) build node features: (x_norm, y_norm) for each pixel
        xs = np.linspace(-self.half_extent + self.cell_size / 2,
                         self.half_extent - self.cell_size / 2,
                         self.grid_size)
        ys = xs.copy()
        xv, yv = np.meshgrid(xs, ys)
        node_xy = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        x_node = torch.from_numpy(node_xy / self.half_extent).float()  # range −1…1

        # ------------------------------------------------------------------
        # 2) labels + mask
        labels = torch.zeros(x_node.size(0), dtype=torch.float32)
        mask = torch.zeros(x_node.size(0), dtype=torch.bool)
        # convert each measurement to grid index
        i = ((df["x"].values - tx_x + self.half_extent) / self.cell_size).astype(int)
        j = ((df["y"].values - tx_y + self.half_extent) / self.cell_size).astype(int)
        valid = (i >= 0) & (i < self.grid_size) & (j >= 0) & (j < self.grid_size)
        node_ids = i[valid] * self.grid_size + j[valid]
        labels[node_ids] = torch.from_numpy(df["rssi"].values[valid]).float()
        mask[node_ids] = True

        # ------------------------------------------------------------------
        # 3) edges & transmitter node
        edge_index_g, edge_attr_g = build_grid_edges(self.grid_size, self.grid_size)
        edge_index_r, edge_attr_r, tx_id = build_ray_edges(
            self.grid_size, self.grid_size, self.cell_size, (0.0, 0.0)
        )

        # concatenate edge sets
        edge_index = torch.cat([edge_index_g, edge_index_r + x_node.size(0)], dim=1)
        edge_attr = torch.cat([edge_attr_g, edge_attr_r], dim=0)

        # ------------------------------------------------------------------
        # 4) add transmitter node to the node feature matrix (all zeros)
        x_node = torch.cat([x_node, torch.zeros(1, x_node.size(1))], dim=0)
        # expand label & mask to keep shape aligned
        labels = torch.cat([labels, torch.zeros(1)])
        mask = torch.cat([mask, torch.zeros(1, dtype=torch.bool)])

        # ------------------------------------------------------------------
        data = Data(
            x=x_node,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            mask=mask,
            scalar=scalar_feats,
        )
        return data

###############################################################################
# Model                                                                       #
###############################################################################

class MLP(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_ch: int, n_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_ch, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        return self.net(x)


class FiLM(nn.Module):
    """Feature‑wise linear modulation (gamma ⊙ x + beta) driven by scalars."""

    def __init__(self, scalar_dim: int, feat_dim: int):
        super().__init__()
        self.mlp = nn.Linear(scalar_dim, feat_dim * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        gamma_beta = self.mlp(s)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * x + beta


# -----------------------------------------------------------------------------
# Graph‑network (MetaLayer) building blocks                                    #
# -----------------------------------------------------------------------------

class EdgeModel(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.mlp = MLP(feat_dim * 2 + 3, feat_dim, feat_dim)

    def forward(self, src, dest, edge_attr, u, batch):  # noqa: D401,E501
        out = torch.cat([src, dest, edge_attr], dim=-1)
        return self.mlp(out)


class NodeModel(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.mlp = MLP(feat_dim * 2, feat_dim, feat_dim)

    def forward(self, x, edge_index, edge_attr, u, batch):  # noqa: D401,E501
        row, col = edge_index
        agg = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, agg], dim=-1)
        return self.mlp(out)


class PropagationGNN(nn.Module):
    def __init__(
        self,
        node_in: int = 2,
        edge_in: int = 3,
        scalar_in: int = 3,
        hidden: int = 128,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.node_enc = MLP(node_in, hidden, hidden)
        self.edge_enc = MLP(edge_in, hidden, hidden)
        self.film = FiLM(scalar_in, hidden)

        self.blocks = nn.ModuleList(
            [MetaLayer(EdgeModel(hidden), NodeModel(hidden), None) for _ in range(depth)]
        )
        self.node_dec = MLP(hidden, hidden, 1)

    # ------------------------------------------------------------------
    def forward(self, data: Data) -> torch.Tensor:  # noqa: D401,E501
        x = self.node_enc(data.x)
        edge_attr = self.edge_enc(data.edge_attr)

        # FiLM conditioning – broadcast scalars to every node in the graph
        if data.scalar.ndim == 1:
            s = data.scalar.unsqueeze(0).expand(x.size(0), -1)
        else:
            s = data.scalar[data.batch]  # type: ignore[attr-defined]
        x = self.film(x, s)

        for blk in self.blocks:
            x, edge_attr, _ = blk(x, data.edge_index, edge_attr, None, None)
        out = self.node_dec(x).squeeze(-1)
        return out

###############################################################################
# Training helpers                                                             #
###############################################################################

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: E501
    return F.mse_loss(pred[mask], target[mask])


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data)
        loss = masked_mse(pred, data.y, data.mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            total_loss += masked_mse(pred, data.y, data.mask).item()
    return total_loss / len(loader)

###############################################################################
# CLI                                                                         #
###############################################################################

def main(argv: Optional[List[str]] = None):  # noqa: D401,E501
    p = argparse.ArgumentParser(description="GNN baseline for RSSI propagation")
    p.add_argument("--data_dir", type=str, help="Directory with per‑cell CSV files")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    ds = RSSIGraphDataset(args.data_dir)
    n_val = max(1, len(ds) // 10)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = PropagationGNN().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        trn = train_epoch(model, train_loader, opt, args.device)
        val = eval_epoch(model, val_loader, args.device)
        print(f"epoch {epoch:02d} | train RMSE {trn**0.5:6.2f} dB | val RMSE {val**0.5:6.2f} dB")

    print("Training finished ✨")


if __name__ == "__main__":
    main()

