#!/usr/bin/env python3
# ------------------------------------------------------------
# localise_client.py  –  Active Bayesian localisation against
#                        RemoteEvaluationEnv (hackathon server)
# ------------------------------------------------------------
import math, argparse, os, random, sys
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull

# ------------- hackathon helpers ----------------------------
from remote_env import RemoteEvaluationEnv          # your snippet
from new_train   import (load_walkable_nodes, make_adjacent_edges,
                         make_ray_edges, pool_mask,
                         RFPredictor)

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected

# ────────────────────────────────────────────────────────────
# Smallest enclosing circle (Welzl, O(k³) here, fine for k≲200)
# ────────────────────────────────────────────────────────────
def enclosing_circle(pts):
    if len(pts) == 1:
        return (*pts[0], 0.0)

    hull = ConvexHull(pts)
    P = pts[hull.vertices]
    best = (None, float("inf"))

    # check pairs first
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            c = (P[i] + P[j]) / 2
            r = np.linalg.norm(P[i] - c)
            if np.all(np.linalg.norm(P - c, axis=1) <= r + 1e-6) and r < best[1]:
                best = (c, r)

    # triples (only if needed)
    if best[0] is None:
        for i in range(len(P)):
            for j in range(i+1, len(P)):
                for k in range(j+1, len(P)):
                    tri = P[[i, j, k]]
                    A = 2 * np.linalg.det(np.c_[tri, np.ones(3)])
                    if abs(A) < 1e-9:
                        continue
                    b = np.sum(tri * tri, 1)
                    cx = np.linalg.det(np.c_[b, tri[:, 1], np.ones(3)]) / A
                    cy = np.linalg.det(np.c_[tri[:, 0], b, np.ones(3)]) / A
                    c = np.array([cx, cy])
                    r = np.max(np.linalg.norm(tri - c, axis=1))
                    if np.all(np.linalg.norm(P - c, axis=1) <= r + 1e-6) and r < best[1]:
                        best = (c, r)
    cx, cy = best[0]
    return float(cx), float(cy), float(best[1])

# ────────────────────────────────────────────────────────────
# Build the *same* graph as in training
# ────────────────────────────────────────────────────────────
def build_graph(mask_path: Path, csv_path: Path, cell_size: int):
    coords, id_map, (Wc, Hc) = load_walkable_nodes(mask_path, cell_size)
    adj = make_adjacent_edges(coords, id_map)

    CACHE = f"train_data/ray_edges_cs{cell_size}.pt"
    if os.path.exists(CACHE):
        ray = torch.load(CACHE)["edge_index"]
    else:
        import pandas as pd
        df = pd.read_csv(csv_path)
        ray = make_ray_edges(df, id_map, cell_size)
        torch.save({"cell_size": cell_size, "edge_index": ray}, CACHE)

    data = HeteroData()
    xy = coords[:, [1, 0]].astype(np.float32)
    xy[:, 0] /= Wc; xy[:, 1] /= Hc
    data["pixel"].x = torch.from_numpy(xy)

    def to_idx(edge_list):
        return to_undirected(torch.tensor(edge_list, dtype=torch.long).t())
    data["pixel", "adjacent", "pixel"].edge_index = to_idx(adj)
    data["pixel", "ray", "pixel"].edge_index = to_idx(ray)
    return data, id_map

# ────────────────────────────────────────────────────────────
# Active Bayesian search (online, via remote env)
# ────────────────────────────────────────────────────────────
def active_localise(model, data, id_map, env,
                    cell_size=4, sigma=5.0, conf=0.95, max_steps=30):
    device = next(model.parameters()).device
    start_obs = env.reset()
    rx_i, rx_j = start_obs["ij"]
    rssi_obs = start_obs["rssi"]

    # posterior over all nodes
    N = data["pixel"].x.size(0)
    tx_ids = torch.arange(N, device=device)
    log_post = torch.full((N,), -math.log(N), device=device)

    # coordinate helper
    def coarse(rc):
        return rc[0] // cell_size, rc[1] // cell_size

    def to_center(rc_cell):
        return (rc_cell[0] * cell_size + cell_size // 2,
                rc_cell[1] * cell_size + cell_size // 2)

    # main loop
    path = []
    rx_r_cell, rx_c_cell = coarse((rx_i, rx_j))

    for step in range(max_steps):
        # --- Bayes update with current measurement ---
        rx_node = id_map.get((rx_r_cell, rx_c_cell))
        if rx_node is None:
            raise RuntimeError("Receiver stepped into non-walkable cell.")

        mu = model(data, tx_ids, torch.tensor([rx_node] * N, device=device))
        log_post += torch.distributions.Normal(mu, sigma).log_prob(
            torch.tensor(rssi_obs, device=device))
        log_post -= torch.logsumexp(log_post, 0)   # renormalise
        path.append(((rx_r_cell, rx_c_cell), rssi_obs))

        # stopping?
        if log_post.exp().max() >= conf:
            break

        # --- choose next move (IG over 4 neighbours) ---
        best_IG, best_move = -1e9, None
        best_rc = None
        for action, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, 1), (0, -1)]):  # N,S,E,W
            nr, nc = rx_r_cell + dr, rx_c_cell + dc
            if (nr, nc) not in id_map:
                continue
            rx_node_cand = id_map[(nr, nc)]
            mu_cand = model(data, tx_ids,
                            torch.tensor([rx_node_cand] * N, device=device))
            dist = torch.distributions.Normal(mu_cand, sigma)
            # MC expected IG
            rsamp = dist.sample((16,))  # [16, N]
            Hcur = -(log_post.exp() * log_post).sum()
            Hafter = []
            for k in range(16):
                lp = log_post + dist.log_prob(rsamp[k])
                post_k = torch.softmax(lp, 0)
                Hafter.append(-(post_k * post_k.log()).sum())
            IG = Hcur - torch.stack(Hafter).mean()
            if IG > best_IG:
                best_IG, best_move, best_rc = IG, action, (nr, nc)

        # --- move + measure ---
        resp = env.step(best_move)
        rx_i, rx_j, rssi_obs = *resp["ij"], resp["rssi"]
        rx_r_cell, rx_c_cell = best_rc

    # --- credible circle (95 % mass) --------------------------
    post = log_post.exp().cpu().numpy()
    # node_id -> center pixel coords
    node_xy = np.array([to_center(rc) for rc, _ in id_map.items()])
    idx_sorted = post.argsort()[::-1]
    cum, keep = 0.0, []
    for idx in idx_sorted:
        cum += post[idx]; keep.append(idx)
        if cum >= conf:
            break
    pts = node_xy[keep]
    cx, cy, R = enclosing_circle(pts)

    # send final circle (use action 0=N as dummy)
    env.step(0, circle=(int(cx), int(cy), float(R)))
    return (cx, cy, R), path

# ────────────────────────────────────────────────────────────
# main glue
# ────────────────────────────────────────────────────────────
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data, id_map = build_graph(args.mask, args.csv, args.cell_size)
    data = data.to(device)

    model = RFPredictor(args.hidden).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=True))
    model.eval()

    env = RemoteEvaluationEnv(team_id=args.team,
                              transmitter_id=args.tx)

    circle, path = active_localise(model, data, id_map, env,
                                   cell_size=args.cell_size,
                                   sigma=5.0, conf=0.95, max_steps=30)
    print(f"Returned circle centre={circle[:2]}  R={circle[2]:.1f} px")
    print(f"Number of steps: {len(path)-1}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Active localisation client")
    ap.add_argument("--team", required=True, help="your hackathon team_id")
    ap.add_argument("--tx",   required=True, help="transmitter id to hunt")
    ap.add_argument("--mask", type=Path, default="walkable_mask.png")
    ap.add_argument("--csv",  type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default="model.pt")
    ap.add_argument("--cell_size", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    main(ap.parse_args())
