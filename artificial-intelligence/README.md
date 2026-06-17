# Artificial Intelligence Project

This project implements a hybrid neuro-symbolic system designed to solve the 8-puzzle (and 15-puzzle) efficiently. It leverages a Feed-Forward Neural Network to learn an informed heuristic function that guides the A* search algorithm, aiming to outperform traditional hand-crafted heuristics like the Manhattan Distance.

## Project Structure and Scripts

### 1. Core Logic
* **`environment.py`**: Manages the game's physical rules, state representations, legal moves generation, and mathematical solvability checks.
* **`search.py`**: Contains the core A* search algorithm implementation, designed flexibly to accept customizable and injected heuristic functions.

### 2. Data Pipeline
* **`generate_data.py`**: Generates a massive dataset of random, solvable initial board configurations and calculates their true optimal cost using the classic A* algorithm.
* **`dataset.py`**: Handles data loading and preprocessing for PyTorch, including the specific one-hot encoding representation of the board states.

### 3. Neural Network
* **`model.py`**: Defines the architecture of the Multi-Layer Perceptron using PyTorch.
* **`train.py`**: Executes the supervised learning loop to train the neural network to predict the optimal cost to the goal from any given board state.

### 4. Evaluation
* **`eval_offline.py`**: Evaluates the standalone neural network on an independent test set to measure its pure prediction accuracy against the true optimal costs.
* **`eval_online.py`**: Evaluates the complete hybrid system (A* guided by the trained neural network) against the baseline classical A* approach, measuring expanded nodes and execution time.

*Authors: Valentino Basili, Giovanni Paolo Maugeri, Akzat Muratbekov*
