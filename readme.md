
# RF Transmitter Foxhunt Challenge

## Challenge Overview

Welcome to the RF Transmitter Localization Challenge! This hackathon task simulates a "foxhunt" scenario in San Francisco. Your objective is to develop a system that can quickly and accurately locate a hidden RF transmitter.

There are lots of potential approaches here. Build a reinforcement learner, build a propagation model, integrate publicly available data - anything you can think of is on the table.

**Scenario:**

*   You control a [SOF operator](https://en.wikipedia.org/wiki/United_States_special_operations_forces) equipped with a man-carried, low-cost, RF sensor.
*   **Constraints:** The sensor lacks high-precision hardware. This means:
    *   No Direction Finding (DF) capabilities.
    *   No precise time synchronization (e.g., Time Difference of Arrival (TDOA) is not feasible).
*   **Capabilities:** The sensor provides a Received Signal Strength Indicator (RSSI) reading at the operator's current location.
*   **Environment:** You operate within an urban environment with known obstacles (provided as a map). Movement is restricted to walkable areas (the white areas on the map).
*   **Goal:** Navigate the environment, taking RSSI measurements at each step, to pinpoint the transmitter's location. You will submit your location estimate as a Circular Error Probable (CEP) - a circle defined by a center (i, j grid coordinates) and a radius (r).

**Winning Condition:** The winning team will be the team that creates the best hunter, which localizes the transmitter most precisely in the fewest number of steps.

$$
score = \left(\sum_{walks} \sum_{steps} 1\right) \cdot \left(\min_{radius \mid \text{circle correct}} \frac{radius}{500} \right) \cdot \left(1 + 5 \sum_{radius \mid \text{circle incorrect}} \frac{radius}{500}\right)
$$

**A score will be produced for each evaluation transmitter your team tries to localize. Each transmitter is scored independently.**

Good luck!

## Rules and Instructions

This repository contains the source code for the simulation environment. Simulation data are hosted at https://tx-hunt.distspec.com.

### Important! Initial Logistics
Please DM Kalman Chapman on Discord (`kalman_70337`) with your team name. He will provide you a key to use for all of your requests to our evaluation server. **Keep this key secret to your team!**

DS is hosting the evaluation server at https://tx-hunt.distspec.com/. You can run the evaluation server locally to test your hunter. **However, before the end of the hackathon, you must run your hunter against ours so that we can score you properly.**

### Training Data

In the `train_data` directory, you can find a png with the map of the area where the hunt will take place.

Using python, you could load this as a boolean array via
```
import numpy as np
from PIL import Image

im = Image.open("./train_data/walkable_mask.png")

walkable_mask = np.asarray(im, dtype=bool) == 0
```
Note that the image uses matrix indexing (starting from the top left), but if we imagine the origin being at the bottom left then the map is oriented with North at the top, and West at the left.

In `training_walks.csv.gz` you can find a number of sample walks with RSSI samples at each position, and the transmitter ground truth position.

Finally, you may make requests against the api with `transmitter_id`:
```
tx0, tx1, tx2, tx3, tx4
```
**These are the same transmitters from the training data, and you are free to evaluate your hunters against them without affecting your score.**

### Test / Evaluation / Scoring

Your hunter will interact with the running API server to navigate the environment and submit location guesses.

There are 5 evaluation transmitters - you may attempt any or all of them. 
Evaluation transmitter ids are
```
tx5, tx6, tx7, tx8, tx9
```
**You are permitted to start as many walks as you desire against the transmitters, and you may walk them concurrently, but every step in each walk is counted in your final score.** Note that this simulates multiple operators starting at the same location, trying to localize the same transmitter, but walking different paths. Feel free to use this to your advantage.

**At any point in the walk, you may submit an estimate of the transmitter location. Your most precise, correct estimate will discount your score. Any correct estimate with a lower precision will not affect your score negatively.**

**Any incorrect estimate - the transmitter is not anywhere within your proposed CEP - will count against you.**

**If you never submit a transmitter location within a walk, the steps will still be counted toward your score.**

**If you never estimate the transmitter position correctly in any walk for a transmitter, you are disqualified from scoring on this transmitter.**

**No proposed transmitter localization radii > 500 will be taken. All submitted radii will be treated as `min(r, 500)`**

### API Details
1.  **Interaction Client:** We provide a Python client class `RemoteEvaluationEnv` in `evaluation_env.py` to simplify interaction with the API. You may use this class in your hunter. Feel free to write your own in any language.

2.  **Workflow:**
    * Make sure that you've received a `team_id`.  
    * Instantiate `RemoteEvaluationEnv` with the API URL (`https://tx-hunt.distspec.com` by default), your assigned `team_id`, and the `transmitter_id` for the current challenge run.
    *   Call `env.reset()` to start a new "walk" (attempt). This returns the initial state: `{"walk_id": int, "ij":(i,j), "xy":(x,y), "rssi": float}`.
    *   Repeatedly call `env.step(action, circle=None)`:
        *   `action`: An integer representing the desired move: `0` (North), `1` (South), `2` (East), `3` (West).
        *   `circle`: (Optional) Your current location estimate as a tuple `(center_i, center_j, radius_r)`. You should refine and submit this guess as your hunter gathers more data.
        *   The `step` method returns the new state after the move: `{"walk_id": int, "ij":(i,j), "xy":(x,y), "rssi": float}`. It will raise a `ValueError` for illegal moves (e.g., into a wall) or `requests.HTTPError` for other API issues.
    *   Your hunter should implement the logic to decide the next `action` based on the history of received `rssi` values and positions, and when/how to update the `circle` guess.

## Optional
Running the server locally - if you want to experiment with the api, feel free. Additionally, feel free to read through the source code to understand how our evaluation server works. However, the test and eval data are only available at the hosted endpoints.

### Prerequisites

*   Docker
*   Docker Compose

### Setup & Running the Environment

The simulation environment is containerized using Docker.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DistributedSpectrum/tx-hunt.git
    cd tx-hunt
    ```

2.  **Configure Environment Variables:**
    The `docker-compose.yml` file uses several environment variables which you will need to set yourself. Consult `docker-compose.yml` for details. A typical `.env` file might look like:

    ```env
    # Database Credentials (defaults are usually fine for local dev)
    DB_USER=postgres
    DB_PASSWORD=postgres
    DB_NAME=hackathon

    # Hackathon Specifics (GET THESE FROM ORGANIZERS)
    TEAM_IDS=team_alpha,team_beta,team_gamma # Example
    TRANSMITTER_IDS=tx_A,tx_B # Example

    # Map/Simulation Parameters (Defaults provided in docker-compose.yml)
    # START_POS=0,0
    # ORIGIN_XY=[0.0,0.0]
    # CELL_SIZE=1.0
    # RETINA_SHAPE=100,100 # Example shape
    ```

3.  **Build and Run:**
    Use Docker Compose to build the images and start the services (API server and PostgreSQL database):
    ```bash
    docker-compose up --build -d
    ```
    The `-d` flag runs the services in detached mode. You can view logs using `docker-compose logs -f`.

4.  **Verify:**
    The API server should be accessible at `http://127.0.0.1:8000`. You can check its health:
    ```bash
    curl http://127.0.0.1:8000/health
    ```
    You should receive `{"status":"ok"}`.

