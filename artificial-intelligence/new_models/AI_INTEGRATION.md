# AI Integration Context: Neural Network Models for 8-Puzzle

This file is designed for AI systems (Copilots, LLMs, Agents) to instantly understand the structure, architecture, execution protocols, and results of the neural network refactoring project located in this directory.

---

## 1. Directory Structure Map

*   **`data/`**: Dataset loaders and generation.
    *   `dataset.py`: Coordinates dataset loader (`PuzzleDataset`) for Positional and Distillation models.
    *   `delta_dataset.py`: Displacement dataset loader (`DeltaPuzzleDataset`) for Delta and Multi-Task models.
    *   `embedding_dataset.py`: ID-based dataset loader (`EmbeddingPuzzleDataset`) for the Embedding model.
    *   `generate_data.py`: Script to generate offline puzzle states and optimal costs.
    *   `datasets/`: Directory containing generated CSV files (`train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`).
*   **`models/`**: Neural network definitions and training scripts.
    *   `positional/`: Model representing absolute tile positions.
    *   `delta/`: Model representing relative tile displacement $(dx, dy)$ from target coordinates.
    *   `embedding/`: Model using `nn.Embedding` lookup layers for tile IDs.
    *   `multitask/`: Multi-Task learning model predicting both search cost and Manhattan Distance.
    *   `small/`: Lightweight model compressed via Knowledge Distillation from the positional model.
*   **`eval/`**: Comparative benchmark scripts and offline/online search metrics.

---

## 2. Model & Input Summary for AI Agents

For quick reference when modifying, generating, or debugging models:

| Model | Input Shape | Input Formula | Architecture Details | Weights File |
| :--- | :--- | :--- | :--- | :--- |
| **Positional** (`PuzzleNet`) | `(batch_size, 18)` | Coordinates $(x,y)$ of tiles $0..8$ | `Linear(18,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)` | `best_puzzle_model.pth` |
| **Delta** (`DeltaPuzzleNet`) | `(batch_size, 18)` | Displacement: $dx = x_{curr} - x_{targ}$, $dy = y_{curr} - y_{targ}$ | `Linear(18,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,16) -> ReLU -> Linear(16,1)` | `best_delta_model.pth` |
| **Embedding** (`EmbeddingPuzzleNet`) | `(batch_size, 9)` | Tile IDs in grid position order | `nn.Embedding(9,16) -> Flatten(144) -> Linear(144,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)` | `best_embedding_model.pth` |
| **Multi-Task** (`MultiTaskDeltaPuzzleNet`) | `(batch_size, 18)` | Spatiotemporal displacements $(dx, dy)$ | Shared: `Linear(18,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,32) -> ReLU`<br>Heads: `cost_head` (Linear(32,1)), `manhattan_head` (Linear(32,1)) | `best_multitask_model.pth` |
| **Small (Distilled)** (`SmallPuzzleNet`) | `(batch_size, 18)` | Coordinates $(x,y)$ of tiles $0..8$ | `Linear(18,48) -> ReLU -> Linear(48,24) -> ReLU -> Linear(24,1)` | `best_small_puzzle_model.pth` |

---

## 3. How to Run (Execution Reference)

### Training Scripts
Run training with optimized batch sizes (GPU/CUDA automatic acceleration):
*   **Positional**: `python new_models/models/positional/train.py` (batch size 256)
*   **Delta**: `python new_models/models/delta/train_delta.py` (batch size 64)
*   **Embedding**: `python new_models/models/embedding/train_embedding.py` (batch size 64)
*   **Multi-Task**: `python new_models/models/multitask/train_multitask.py` (batch size 64)
*   **Knowledge Distillation**: `python new_models/models/small/train_distill.py` (batch size 256, requires trained positional model)

### Evaluation Scripts
Verify and benchmark models:
*   **Full Comparative Benchmark**: `python new_models/eval/eval_embedding.py`
*   **Teacher vs Student KD**: `python new_models/eval/eval_all.py`
*   **Offline MAE Metric Summary**: `python new_models/eval/eval_metrics.py`

---

## 4. Path Management / Import Resolution (CRITICAL)

All scripts inside `new_models/` use a dynamic `sys.path` patch to allow execution from either the repository root or subfolders, and to import files (`environment.py`, `search.py`) from the parent root directory.

Ensure any new Python script includes this block at the very top:

```python
import os
import sys

# Dynamic lookup of new_models and parent root directory
_current = os.path.abspath(__file__)
while True:
    _parent = os.path.dirname(_current)
    if _parent == _current:
        break
    if os.path.basename(_parent) == 'new_models':
        sys.path.append(_parent)
        sys.path.append(os.path.dirname(_parent))
        root_dir = _parent
        break
    _current = _parent
```

`root_dir` will point to the `new_models` folder path to load weights and datasets properly.

---

## 5. Benchmarks & Performance Baselines

Average metrics recorded on a test set of 1,000 samples and A* search on 10 configurations:

*   **Classic A* (Manhattan)**: $1039.7$ nodes expanded | $0.009 \text{ s}$ execution.
*   **Delta Model**: **$64.6$ nodes expanded** | **$0.080 \text{ s}$** execution (**State-Of-The-Art**).
*   **Multi-Task Delta**: $69.3$ nodes expanded | $0.087 \text{ s}$ execution.
*   **Embedding Model**: $114.5$ nodes expanded | $0.098 \text{ s}$ execution (**Best Offline MAE: 1.460**).
*   **Positional**: $184.5$ nodes expanded | $0.227 \text{ s}$ execution.
*   **Small Student (KD)**: $372.3$ nodes expanded | $0.374 \text{ s}$ execution.
