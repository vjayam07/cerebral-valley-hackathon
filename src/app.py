import os, json
from typing import Tuple, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import asyncpg
import requests

app = FastAPI(debug=True)

# ── Load config & maps ───────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "hackathon"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}
TRANSMITTER_IDS = os.getenv("TRANSMITTER_IDS", "").split(",")
START_I, START_J = tuple(map(int, os.getenv("START_POS", "0,0").split(",")))
shape = tuple(map(int, os.getenv("RETINA_SHAPE", "0,0").split(",")))
origin_xy = np.array(json.loads(os.getenv("ORIGIN_XY", "[0.0,0.0]")))
cell_size = float(os.getenv("CELL_SIZE", "1.0"))

walkable = np.fromfile(os.getenv("WALKABLE_MASK_PATH", "data/walkable_mask.npy"), dtype=bool).reshape(shape)
rssi_maps = {
    tx: np.fromfile(f"data/rssi_tx_{tx[-1]}.npy", dtype=np.float32).reshape(shape)
    for tx in TRANSMITTER_IDS
}
Nx, Ny = walkable.shape

db_pool: asyncpg.pool.Pool

# ── Startup ──
@app.on_event("startup")
async def startup():
    if os.getenv("UVICORN_WORKER_ID", "0") != "0":
        return
    global db_pool
    db_pool = await asyncpg.create_pool(**DB_CONFIG)
    async with db_pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id TEXT PRIMARY KEY
        );
        """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS walks (
            team_id TEXT NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
            walk_id INTEGER NOT NULL,
            transmitter_id TEXT NOT NULL,
            current_i INTEGER NOT NULL,
            current_j INTEGER NOT NULL,
            PRIMARY KEY (team_id, walk_id, transmitter_id)
        );
        """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            step_id SERIAL PRIMARY KEY,
            team_id TEXT NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
            walk_id INTEGER NOT NULL,
            transmitter_id TEXT NOT NULL,
            step_no INTEGER NOT NULL,
            action INTEGER NOT NULL,
            i INTEGER NOT NULL,
            j INTEGER NOT NULL,
            rssi REAL NOT NULL,
            locale_i INTEGER,
            locale_j INTEGER,
            locale_r REAL,
            ts TIMESTAMPTZ DEFAULT now()
        );
        """)

@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()

# ── Pydantic models ─────────────────────────────────────────────────────────
class LocaleGuess(BaseModel):
    i: int
    j: int
    r: float

class StartWalk(BaseModel):
    team_id: str
    walk_id: int
    transmitter_id: str

class StepIn(BaseModel):
    team_id: str
    walk_id: int
    transmitter_id: str
    action: int  # 0=N,1=S,2=W,3=E
    locale: Optional[LocaleGuess] = None

# ── Public endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/start")
async def start_walk(req: StartWalk):
    async with db_pool.acquire() as conn:
        exists = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM teams WHERE team_id=$1)", req.team_id)
        if not exists:
            raise HTTPException(400, "unknown team")

        if not walkable[START_I, START_J]:
            raise HTTPException(500, "start pos not walkable")

        if not req.transmitter_id in TRANSMITTER_IDS:
            raise HTTPException(400, "unknown transmitter")

        try:
            await conn.execute(
                "INSERT INTO walks VALUES ($1, $2, $3, $4, $5)",
                req.team_id, req.walk_id, req.transmitter_id, START_I, START_J
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(400, "walk already exists")

        rssi = float(np.nan_to_num(rssi_maps[req.transmitter_id][START_J, START_I] + (3 * np.random.randn())))
        await conn.execute(
            "INSERT INTO steps (team_id, walk_id, transmitter_id, step_no, action, i, j, rssi) VALUES ($1,$2,$3,0,-1,$4,$5,$6)",
            req.team_id, req.walk_id, req.transmitter_id, START_I, START_J, rssi
        )

    return {"ij": (START_I, START_J), "rssi": rssi}

@app.post("/step")
async def step(req: StepIn):
    async with db_pool.acquire() as conn:
        exists = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM teams WHERE team_id=$1)", req.team_id)
        if not exists:
            raise HTTPException(400, "unknown team")

        if not req.transmitter_id in TRANSMITTER_IDS:
            raise HTTPException(400, "unknown transmitter")

        row = await conn.fetchrow(
            "SELECT current_i, current_j FROM walks WHERE team_id=$1 AND walk_id=$2 AND transmitter_id=$3",
            req.team_id, req.walk_id, req.transmitter_id
        )
        if not row:
            raise HTTPException(400, "walk not found")

        ci, cj = row["current_i"], row["current_j"]
        delta = {3: (-1, 0), 2: (1, 0), 1: (0, -1), 0: (0, 1)}.get(req.action)
        if delta is None:
            raise HTTPException(400, "bad action")

        ni, nj = ci + delta[0], cj + delta[1]
        if not (0 <= ni < Nx and 0 <= nj < Ny and walkable[nj, ni]):
            raise HTTPException(400, "illegal_move")

        rssi = float(np.nan_to_num(rssi_maps[req.transmitter_id][nj, ni] + (3 * np.random.randn())))
        await conn.execute(
            "UPDATE walks SET current_i=$1, current_j=$2 WHERE team_id=$3 AND walk_id=$4 AND transmitter_id=$5",
            ni, nj, req.team_id, req.walk_id, req.transmitter_id
        )

        rec = await conn.fetchrow(
            "SELECT COALESCE(MAX(step_no), 0) + 1 AS coalesce FROM steps WHERE team_id=$1 AND walk_id=$2 AND transmitter_id=$3",
            req.team_id, req.walk_id, req.transmitter_id
        )
        step_no = rec["coalesce"]

        await conn.execute(
            "INSERT INTO steps (team_id, walk_id, transmitter_id, step_no, action, i, j, rssi, locale_i, locale_j, locale_r) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            req.team_id, req.walk_id, req.transmitter_id,
            step_no, req.action, ni, nj, rssi,
            req.locale.i if req.locale else None,
            req.locale.j if req.locale else None,
            req.locale.r if req.locale else None
        )

    response = {"ij": (ni, nj), "rssi": rssi}
    if req.locale:
        response["locale"] = req.locale.dict()
    return response
