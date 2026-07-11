# DAVI Module heuristic learner


## 1. Module identity

- **Name**: DAVI (Deep Approximate Value Iteration) heuristic learner
- **Location**: `artificial-intelligence/davi/`
- **Parent project**: `artificial-intelligence/` — a hybrid neuro-symbolic
  8-puzzle solver (neural-network heuristic + A* search)
- **Relationship to parent project**: additive alternative training method to
  the supervised pipeline in the project root (`model.py`/`train.py`); does
  not modify it; reuses `environment.py` and `search.py` from the root
  unchanged
- **Scope**: 8-puzzle (3x3) only. Not applied to the 15-puzzle in current code
- **Status**: implemented, trained, evaluated

## 2. Problem statement

- Puzzle: 3x3 sliding-tile board. State = tuple of 9 ints, `0` = blank.
  Goal state = `(1,2,3,4,5,6,7,8,0)`.
- Objective: find the shortest move sequence from a scrambled state to the
  goal state.
- Search algorithm: A* (`search.py`, `f = g + h`), shared/unchanged from
  project root.
- Heuristic's role: `h(board)` estimates moves remaining. A tighter, safer
  (non-overestimating) `h` → fewer nodes expanded by A*, while preserving
  optimality.
- Baseline heuristic: Manhattan distance — admissible (provably never
  overestimates), hand-written, zero training cost.

## 3. Approach: DAVI (Deep Approximate Value Iteration)

- **Source**: Agostinelli, McAleer, Shmakov, Baldi (2019), *"Solving the
  Rubik's Cube with Deep Reinforcement Learning and Search"* (DeepCubeA),
  Nature Machine Intelligence. https://doi.org/10.1038/s42256-019-0070-z
- **Learning category**: self-supervised / bootstrapped reinforcement
  learning — specifically approximate value iteration with a neural function
  approximator. Not supervised learning (no external labeled dataset) and not
  unsupervised learning (there is still a numeric regression target per
  example).
- **Key property**: requires no pre-solved dataset and no solver call during
  training — unlike the project root's supervised pipeline, which runs A* on
  thousands of boards up front to generate training labels.
- **Training data generation**: "backward scrambling" — start at the goal
  state, take `N` random legal moves backward (`N` sampled uniformly from
  `1..MAX_SCRAMBLE` per generated board). Always yields a solvable board, no
  solver needed.
- **Training target (label)**: Bellman backup.
  - `J(goal) = 0` — hardcoded constant, never produced by a network.
  - `J(s) = 1 + min(J(neighbor) for neighbor in neighbors(s))` for all other
    `s`, where neighbor values come from a frozen target network.
- **Convergence mechanism**: value iteration. The one anchored fact
  (`J(goal)=0`) propagates outward through repeated Bellman backups —
  provably convergent with an exact per-state lookup table; only
  *approximate* here because a neural network generalizes across states
  instead of memoizing each one individually.
- **Two-network training scheme**:
  - `model`: the live network; the only one that receives gradients
    (`MSELoss` + Adam optimizer).
  - `target_net`: a frozen copy of `model`; supplies labels only, never
    trained directly; hard-updated (`load_state_dict`, a plain weight copy,
    not a training step) every `SYNC_EVERY` iterations.
  - Purpose: prevents training against a self-referential target that shifts
    at the same instant it's being fitted (instability). Same "target
    network" mechanism used in DQN (Mnih et al., 2015).
  - Note: `target_net`'s outputs are only accurate near the goal early in
    training; the region of accuracy expands outward with each sync — it is
    not a static, permanently-incorrect function.

## 4. Architecture: `PuzzleResNet` (`davi_model.py`)

- **Input**: 18 floats — `(row, col)` per tile, ordered by tile value (tile
  0's row/col first, tile 1's next, ...). Identical encoding scheme to the
  project root's `dataset.py`/`model.py`.
- **Layer stack**:
  1. `Linear(18 → 256)` + `BatchNorm1d(256)` + `ReLU`
  2. 4x `ResidualBlock(dim=256)`: `Linear(256→256)+BN+ReLU` → `Linear(256→256)+BN`
     → add original block input (skip connection) → `ReLU`
  3. `Linear(256 → 1)` — scalar output, estimated moves remaining
- **Parameter count**: 536,065 (measured)
- **Design rationale**: residual/skip connections + BatchNorm chosen to keep
  a deeper network trainable/stable. The exact size (4 blocks, width 256) is
  an empirical choice, not analytically derived — comparatively large for the
  8-puzzle's state space (see Limitations, §8).
- **Comparison point**: the project root's supervised model
  (`model.py:PuzzleNet`) is a plain MLP, `18→128→64→32→1`, ReLU only, no
  BatchNorm/residual connections, 12,801 parameters (~42x smaller than
  `PuzzleResNet`).

## 5. Hyperparameters (`train_davi.py`)

| Name | Value | Meaning |
|---|---|---|
| `ITERATIONS` | 2000 | total training steps (not epochs — no fixed dataset; a fresh random batch is generated every step) |
| `BATCH_SIZE` | 1000 | boards generated per iteration |
| `MAX_SCRAMBLE` | 30 | max backward-scramble depth (the 8-puzzle's true maximum optimal solution length is 31) |
| `LEARNING_RATE` | 0.001 | Adam optimizer |
| `SYNC_EVERY` | 50 | iterations between target-network hard updates (40 total syncs over the full run) |
| Loss function | `MSELoss` | between `model`'s prediction and the Bellman-backup target |
| Total board samples processed | 2,000,000 | `ITERATIONS × BATCH_SIZE` |

## 6. File map

| File | Contents |
|---|---|
| `davi_model.py` | `PuzzleResNet` + `ResidualBlock` class definitions |
| `davi_utils.py` | `encode_state`/`encode_states` (board → tensor), `scramble_from_goal` (backward scrambler), `compute_bellman_targets` (label generator using `target_net`), `NeuralHeuristic` (A* heuristic wrapper, with caching, clamps output ≥ 0), `save_loss_plot` |
| `train_davi.py` | training loop entry point; saves `davi_model.pth` and `davi_loss_curve.png` |
| `eval_online.py` | loads `davi_model.pth`, runs A* with the DAVI heuristic vs. Manhattan on 100 random solvable boards, reports nodes expanded / time / solution length / optimality |
| `davi_model.pth` | trained weights (2.18 MB), consistent with `PuzzleResNet`'s parameter count |

Shared dependencies from project root (unmodified): `environment.py` (board
mechanics — `get_neighbors`, `is_solvable`, `generate_solvable_state`,
`GOAL_STATE_8`, `GRID_SIZE_8`) and `search.py` (`a_star`, `manhattan_distance`
— note `GOAL_STATE`/`GRID_SIZE`/`TARGET_POS` are module-level globals
hardcoded to the 8-puzzle configuration in that file, not parameters).

## 7. Measured performance

All numbers below were measured directly (not simulated), comparing DAVI
against Manhattan and against the project root's supervised MLP
(`best_puzzle_model.pth`), on identical boards.

### 7a. Offline accuracy
1000 boards from the project root's `test_dataset.csv` (A*-labeled ground
truth; these boards were never seen during DAVI training, which only uses
self-generated scrambles).

| Metric | Supervised MLP | DAVI ResNet |
|---|---|---|
| Mean Absolute Error | 1.60 moves | 6.26 moves |
| Overestimation rate | 45.8% (458/1000) | 4.8% (48/1000) |
| Error, easy boards (0–10 moves, n=3) | 2.31 | 0.22 |
| Error, medium boards (11–20 moves, n=316) | 1.80 | 2.67 |
| Error, hard boards (21–31 moves, n=681) | 1.51 | 7.96 |
| Inference throughput | 41,327 pred/sec | 7,203 pred/sec |

### 7b. Online A* search
100 boards, unweighted A* (`weight_factor=1`), ground truth = CSV optimal
cost (independent of Manhattan).

| Method | Avg. nodes expanded | Avg. solve time | Optimal rate |
|---|---|---|---|
| A* + Manhattan (baseline) | 1768.8 | 6.5 ms | 100% (proven admissible) |
| A* + Supervised MLP | 4773.3 | 86.4 ms | 100% (observed) |
| A* + DAVI ResNet | 994.2 | 159.6 ms | 100% (observed) |

### 7c. Key interpretation (factual, not opinion)

- DAVI reduces node expansions vs. Manhattan by ~1.78x while remaining
  optimal on every board tested.
- The supervised MLP, despite lower average prediction error (MAE), expands
  *more* nodes than Manhattan when used unweighted. Attributable to its much
  higher overestimation rate (45.8% vs. DAVI's 4.8%) — overestimation rate,
  not raw MAE, is the determining factor for a heuristic's effect on A*
  search efficiency.
- DAVI is slower in wall-clock time than both Manhattan and the supervised
  MLP despite fewer node expansions, because `PuzzleResNet`'s forward pass
  (536K params) costs more per call than `PuzzleNet`'s (12.8K params) or
  Manhattan's plain arithmetic.

## 8. Limitations (explicit, factual)

- No formal admissibility/optimality guarantee (Manhattan's is a mathematical
  proof; DAVI's optimality is an empirical observation on tested samples
  only).
- Overestimates on 4.8% of an independent 1000-board test set.
- Slower wall-clock time than Manhattan on this puzzle size — network
  inference cost outweighs the search-efficiency gain at this scale.
- Network is likely oversized (536K params) relative to the 8-puzzle's state
  space (~181,000 reachable boards).
- Prediction error increases substantially with true distance from goal
  (hardest boards: ~8 moves average error).
- Hardcoded to the 8-puzzle (`GRID_SIZE_8` fixed in `davi_model.py`; the
  shared `search.py` also hardcodes its goal/grid-size globals to the
  8-puzzle configuration).
- No batched heuristic evaluation in `search.py`'s A* (heuristic is called
  once per node, not vectorized) — a likely bottleneck if scaled up.

## 9. Potential extensions (not implemented)

- **Scaling to the 15-puzzle or larger**: the training method (backward
  scrambling + Bellman bootstrap) needs no solver and is architecture-agnostic
  to puzzle size in principle. Would require: parameterizing
  `input_size`/`grid_size` in `davi_model.py` and `davi_utils.py`, a
  puzzle-size-generic A* implementation (current `search.py` is hardcoded to
  one configuration at a time), a larger network, more training iterations,
  and likely batched heuristic evaluation plus weighted A*/IDA* to stay
  tractable at scale.
- **Smaller-network experiment**: could plausibly reduce wall-clock time
  while retaining most of the node-expansion benefit; untested.
- **Curriculum/prioritized training by scramble depth**: a plausible way to
  reduce the hard-board error specifically; not implemented (current
  training samples scramble depth uniformly at random, 1–30, every batch).

## 10. Reference

Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). *Solving the
Rubik's Cube with Deep Reinforcement Learning and Search"* (DeepCubeA), https://doi.org/10.1038/s42256-019-0070-z
