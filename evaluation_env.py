from typing import Dict, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter

class RemoteEvaluationEnv:
    """Wraps the hackathon API so participants can code against identical API.
    Each reset() generates a new walk_id, and step() automatically uses the current walk."""
    def __init__(
        self,
        team_id: str,
        transmitter_id: str,
        base_url: str = "https://tx-hunt.distspec.com",
    ):
        self.base = base_url.rstrip("/")
        self.team = team_id
        self.tx = transmitter_id
        self._walk_counter: int = 0
        self.current_walk_id: Optional[int] = None

        # --- NEW: one Session for all HTTP calls ---
        self.session = requests.Session()
        # tune your pool sizes if you need higher concurrency
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=50)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        # ensure keep-alive (requests does this by default, but explicit is OK)
        self.session.headers.update({"Connection": "keep-alive"})

    def reset(self) -> Dict:
        """
        Starts a fresh walk: increments walk_id, calls /start, and returns:
        {"walk_id": int, "ij":(i,j), "xy":(x,y), "rssi": rssi}
        """
        self._walk_counter += 1
        self.current_walk_id = self._walk_counter
        payload = {
            "team_id": self.team,
            "walk_id": self.current_walk_id,
            "transmitter_id": self.tx,
        }

        resp = self.session.post(f"{self.base}/start", json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        data["walk_id"] = self.current_walk_id
        return data

    def step(self, action: int, circle: Optional[Tuple[int, int, float]] = None) -> Dict:
        """
        Sends one move in the current walk_id.
        Actions are: 0=N, 1=S, 2=E, 3=W.
        Optionally includes a localization guess: (i, j, r) as a circle.
        Returns: {"walk_id": int, "ij":(i,j), "xy":(x,y), "rssi": rssi}
        """
        if self.current_walk_id is None:
            raise RuntimeError("Call reset() before step()")

        payload = {
            "team_id": self.team,
            "walk_id": self.current_walk_id,
            "transmitter_id": self.tx,
            "action": action,
        }
        if circle:
            payload["locale"] = {"i": circle[0], "j": circle[1], "r": circle[2]}

        resp = self.session.post(f"{self.base}/step", json=payload, timeout=5)
        if resp.status_code == 403:
            raise ValueError(f"Illegal move in walk {self.current_walk_id}")
        resp.raise_for_status()

        data = resp.json()
        data["walk_id"] = self.current_walk_id
        return data
