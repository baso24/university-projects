# Technical Report: Neural Network Models for the 8-Puzzle

This documentation describes the Deep Learning approaches implemented to learn and approximate the heuristic function of the A* search algorithm for the 8-Puzzle game.

---

## 1. Overview & Rationale (Why these directions?)

The A* search algorithm relies heavily on a heuristic function $h(n)$ to estimate the minimum cost to reach the goal state. Classic heuristics, such as the *Manhattan Distance*, are computationally fast to evaluate mathematically but ignore tile collisions, expanding a large number of redundant nodes on complex board configurations.

The core idea of this work is to **replace or enhance the classic heuristic with a neural network regressor** trained to approximate the actual optimal path cost (computed offline via exhaustive search).

During development, we focused on three main research pillars:
1. **Input Representation (Feature Engineering)**: Moving from absolute coordinates to relative spatial displacements (*Delta*) and embeddings to facilitate the network's understanding of spatial relationships.
2. **Multi-Task Learning**: Introducing an inductive bias by forcing the network to simultaneously learn both the true search cost (hard target) and the Manhattan Distance (easy but highly correlated target).
3. **Model Compression (Knowledge Distillation)**: Mitigating CPU inference latency during A* search by distilling knowledge from a larger pre-trained network ("Teacher") to a much lighter and faster network ("Student").

---

## 2. Model Architecture Details

```
               +-------------+
               | Board State |
               +------+------+
                      |
      +---------------+---------------+
      |               |               |
      v               v               v
 [Coordinates]    [Delta (dx,dy)]  [Tile IDs]
      |               |               |
      v               +-------+       v
(PuzzleNet)           |       |  (EmbeddingPuzzleNet)
      :               v       v
(Distillation)   (DeltaNet) (MultiTaskNet)
      :
(SmallPuzzleNet)
```

---

### A. Positional Model (`PuzzleNet`)
The baseline model that learns the heuristic directly from the absolute spatial positions of the tiles on the board.

*   **Input Format**:
    *   Representation: A flat vector of size $18$ ($9 \text{ tiles} \times 2 \text{ coordinates}$).
    *   Calculation: For each tile value $v \in \{0..8\}$ (where 0 represents the blank space), we calculate its 2D coordinates `(row, col)` in the current board. The vector is ordered by tile value: the first two elements are the coordinates of tile 0, the next two of tile 1, and so on.
*   **Architecture & Layers**:
    *   Standard Multi-Layer Perceptron (MLP).
    *   Layers: `Linear(18, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)`.
*   **Rationale**:
    *   Providing $(x, y)$ coordinates instead of raw board indices prevents the network from treating adjacent cells (e.g., indices 2 and 3) as distant, providing a 2D geometric inductive bias.

---

### B. Delta Model (`DeltaPuzzleNet`)
An approach based on the relative distance of each tile from its target coordinate in the goal state.

*   **Input Format**:
    *   Representation: A flat vector of size $18$ ($9 \text{ tiles} \times 2 \text{ displacements}$).
    *   Calculation: For each tile $v$ currently positioned at `(curr_x, curr_y)`, we calculate its 2D displacement relative to its target position `(targ_x, targ_y)` in the Goal state:
        $$dx = curr\_x - targ\_x$$
        $$dy = curr\_y - targ\_y$$
        The vector stores $[dx_0, dy_0, dx_1, dy_1, \dots, dx_8, dy_8]$.
*   **Architecture & Layers**:
    *   Lightweight MLP. Given the highly informative, pre-engineered input features, the hidden layers are reduced in size.
    *   Layers: `Linear(18, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)`.
*   **Rationale**:
    *   Instead of forcing the network to memorize the goal positions and compute distances internally, we feed the spatial discrepancy directly. This simplifies the regression task, allowing the model to focus on tile interactions and local collisions.

---

### C. Embedding Model (`EmbeddingPuzzleNet`)
An approach that treats the board state as a sequence of discrete categories, projecting the tile values into a continuous latent space.

*   **Input Format**:
    *   Representation: A flat vector of $9$ integer IDs (`torch.long`).
    *   Calculation: The vector represents the board tiles read in grid order (top-left to bottom-right), e.g., `[1, 3, 0, 4, 2, 5, 7, 8, 6]`.
*   **Architecture & Advanced Techniques**:
    *   **Embedding Layer**: The network starts with an embedding layer (`nn.Embedding(9, 16)`). This maps each of the 9 discrete tile IDs to a 16-dimensional continuous vector (learned during training).
    *   The 9 vectors are flattened into a single input representation of size $144$ ($9 \times 16$).
    *   Subsequent MLP layers: `Linear(144, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)`.
*   **Rationale**:
    *   Using raw tile integers implies a false numerical scale (e.g., tile 8 is not "larger" or "more important" than tile 1). The embedding layer allows the network to learn a dense continuous vector for each tile value, clustering tiles with similar behaviors or physical constraints.

---

### D. Multi-Task Model (`MultiTaskDeltaPuzzleNet`)
A model that leverages multi-objective learning to guide the network's shared internal representations.

*   **Input Format**:
    *   Relative spatial displacements $(dx, dy)$ for each tile (size $18$), identical to the Delta model.
*   **Architecture & Advanced Techniques**:
    *   **Shared Backbone**: A shared MLP block: `Linear(18, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 32) -> ReLU`.
    *   **Multi-Head Output**: Two parallel linear heads connected to the shared block:
        1.  `cost_head` (Linear(32, 1)): Predicts the true optimal path cost (primary target).
        2.  `manhattan_head` (Linear(32, 1)): Approximates the Manhattan Distance of the board (auxiliary target).
    *   **Combined Loss Function**: During training, the loss is a weighted sum:
        $$\mathcal{L}_{total} = \mathcal{L}_{MSE}(Cost) + \alpha \cdot \mathcal{L}_{MSE}(Manhattan)$$
        (with $\alpha = 0.5$).
*   **Rationale**:
    *   Predicting the true cost is a highly non-convex, complex problem. The Manhattan Distance is a simple, linear, and highly correlated heuristic. Forcing the shared layers to simultaneously predict both acts as a spatial regularizer, reducing overfitting on the training set and improving true cost generalization.

---

### E. Student Model (`SmallPuzzleNet` via Knowledge Distillation)
An extremely compact model designed to maximize inference speed.

*   **Input Format**:
    *   Absolute coordinate vector of size $18$, identical to the `PuzzleNet` Teacher.
*   **Architecture & Advanced Techniques**:
    *   Ultralight MLP: `Linear(18, 48) -> ReLU -> Linear(48, 24) -> ReLU -> Linear(24, 1)`.
    *   **Knowledge Distillation (KD)**: Trained using both true costs (*hard labels*) and teacher predictions (*soft labels*). The loss function is:
        $$\mathcal{L}_{KD} = \beta \cdot \mathcal{L}_{MSE}(y_{pred}, y_{true}) + (1 - \beta) \cdot \mathcal{L}_{MSE}(y_{pred}, y_{teacher})$$
        (with $\beta = 0.5$).
*   **Rationale**:
    *   In A* search, network propagation time on CPU is often the main bottleneck. A large model reduces node expansions but increases overall search time. Knowledge distillation transfers the generalization capability of the Teacher to a network with less than 1/3 of the parameters, optimizing the search time trade-off.

---

## 3. Experimental Results & Comparisons

All models were evaluated on a benchmark suite of test board configurations. The average metrics of the fully trained models are reported below:

| Heuristic Model | Mean Absolute Error (MAE) | Average Nodes Expanded | Average A* Search Time | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Classic A* (Manhattan)** | *N/A* | $1039.7$ | **$0.009 \text{ s}$** | Extremely fast computations, but expands too many nodes. |
| **Delta Model (`DeltaPuzzleNet`)** | $1.697$ | **$64.6$** | **$0.080 \text{ s}$** | **Best Performance Overall**: Relative displacement significantly simplifies search. |
| **Multi-Task Delta** | $1.629$ | $69.3$ | $0.087 \text{ s}$ | Excellent performance, closely aligned with Delta due to Manhattan regularizer. |
| **Embedding Model** | **$1.460$** | $114.5$ | $0.098 \text{ s}$ | **Lowest Offline Estimation Error**: Embedding successfully learned spatial relations. |
| **Positional Teacher (`PuzzleNet`)** | $1.959$ | $184.5$ | $0.227 \text{ s}$ | Significant node reduction (82%), but slowed down by inference complexity. |
| **Small Student (`SmallPuzzleNet`)** | *N/A* | $372.3$ | $0.374 \text{ s}$ | 64% node reduction compared to Manhattan, but suffers from spatial compression. |

---

## 4. Critical Discussion & Conclusions

1.  **The Power of Feature Engineering (Delta Models)**:
    Relative tile displacement $(dx, dy)$ is the most effective feature representation. By directly supplying the spatial discrepancy relative to the Goal coordinates, the **Delta** model reduces the search space by **93.8%** (only 64.6 nodes expanded on average) and achieves the fastest search time (0.080 s) due to its simple network structure.
2.  **The Success of Embeddings**:
    Unlike initial trials with incomplete training, the fully trained **Embedding Model** (100 epochs, batch size 64) achieved the **lowest Mean Absolute Error (MAE = 1.460) offline**. The continuous representations learned by the `nn.Embedding(9, 16)` layer allowed the network to map physical tile configurations with extreme precision, expanding only 114.5 nodes.
3.  **The Neural Inference Trade-off**:
    Although neural networks drastically reduce search node expansion (up to 93.8%), their actual execution time in A* is higher than the classic Manhattan Distance calculated in pure procedural code. However, in applications where state evaluation is naturally expensive (e.g., robotics, physics simulations), the node savings provided by models like **Delta** and **Multi-Task** provide a crucial, absolute advantage.
