<div align="center">

# 🎓 University Projects Showcase
### Master's Degree in Artificial Intelligence & Automation Engineering
**Università di Siena** • *A.Y. 2025/2026*

<br>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![OpenMP](https://img.shields.io/badge/OpenMP-FFA500?style=for-the-badge&logo=c%2B%2B&logoColor=black)](https://www.openmp.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-ED8B00?style=for-the-badge&logo=mathworks&logoColor=white)](https://mathworks.com)
[![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org)

</div>

---

## 📌 Overview

| Subject | Project Title | Key Tech | Quick Link |
| :--- | :--- | :--- | :---: |
| **🖼️ Digital Image Processing** | Pose Estimation and Fall Detection | `YOLO26-seg` `Python` `OpenCV` | [Explore](./digital-image-processing) |
| **🤖 Artificial Intelligence** | Hybrid Neurosymbolic System for 15-Puzzle | `Python` `PyTorch` `A* Search` | [Explore](./artificial-intelligence) |
| **🧠 Neural Networks** | Conditional GAN on CelebA with GUI | `PyTorch` `cGAN` `DCGAN` `GUI` | [Explore](./neural-networks) |
| **⚡ High Performance Architecture** | Parallel Sparse Matrix Computation | `CUDA` `OpenMP` `C++` | [Explore](./high-performance-computer-architecture) |
| **📊 Fundamentals of Machine Learning** | XOR Network & Decision Boundaries | `NumPy` `Optimization` `Matplotlib` | [Explore](./fundamentals-of-machine-learning) |
| **⏱️ Discrete Event Systems** | D.E.S. Model of a Train Station | `MATLAB` `Markov Chains` `Automata` | [Explore](./discrete-event-systems) |

---

## 🚀 Projects

### 🖼️ Digital Image Processing
**Title:** Pose Estimation and Fall Detection  
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square) ![YOLO-seg](https://img.shields.io/badge/YOLO-Seg-brightgreen?style=flat-square) ![Computer Vision](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat-square)

> Implementation of a custom training pipeline for a YOLO segmentation network focused only on specific body parts (head, torso, arms, legs, and feet). The spatial relationships between body keypoints are analyzed via kinematic vectors and Mahalanobis distance scoring to robustly determine whether a human fall has occurred in video sequences.

🔗 **Links:** [📁 Browse Directory](./digital-image-processing) • [📖 Detailed Project README](./digital-image-processing/README.md)

<br>

### 🤖 Artificial Intelligence
**Title:** Hybrid neurosymbolic system for solving the 15-puzzle  
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square) ![A* Search](https://img.shields.io/badge/Search-A*-orange?style=flat-square)

> Design and implementation of a hybrid neuro-symbolic system designed to solve the 15-puzzle. A deep feed-forward neural network is trained via supervised learning to predict the optimal remaining cost from any board state, serving as an informed heuristic function that guides the classic A* search algorithm beyond standard Manhattan distance baselines.

🔗 **Links:** [📁 Browse Directory](./artificial-intelligence) • [📖 Detailed Project README](./artificial-intelligence/README.md)

<br>

### 🧠 Neural Networks
**Title:** Conditional GAN on CelebA dataset with GUI  
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square) ![cGAN](https://img.shields.io/badge/Architecture-cGAN%20%2F%20DCGAN-green?style=flat-square) ![GUI](https://img.shields.io/badge/UI-Interactive%20GUI-purple?style=flat-square)

> Training, implementation, and evaluation of Conditional Generative Adversarial Networks (cGAN and DCGAN) on the CelebA dataset. Includes an interactive graphical user interface (GUI) enabling users to control and synthesize facial images based on customizable facial attribute selections.

🔗 **Links:** [📁 Browse Directory](./neural-networks)

<br>

### ⚡ High Performance Computer Architecture
**Title:** Analysis and testing of Chapter 10 (Sparse Matrix Computation) — *Programming Massively Parallel Processors (3rd Ed.)*  
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=flat-square) ![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-yellow?style=flat-square) ![C++](https://img.shields.io/badge/C++-High%20Performance-00599C?style=flat-square)

> Implementation and experimental evaluation of parallel sparse matrix algorithms (SpMV). Compares high-performance pure CUDA kernel implementations, OpenMP multithreaded implementations, and CPU sequential baselines to analyze the acceleration and memory efficiency gained through parallel computing.

🔗 **Links:** [📁 Browse Directory](./high-performance-computer-architecture) • [📄 Project Report (PDF)](./high-performance-computer-architecture/HPCA__1_.pdf)

<br>

### 📊 Fundamentals of Machine Learning
**Title:** XOR network  
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square) ![Machine Learning](https://img.shields.io/badge/ML-From%20Scratch-success?style=flat-square)

> Implementation of a neural network to solve the classic non-linear XOR classification problem from scratch. Features interactive visualizations illustrating the progressive transformation of decision separation surfaces during gradient descent learning and the evolution of the empirical risk function.

🔗 **Links:** [📁 Browse Directory](./fundamentals-of-machine-learning)

<br>

### ⏱️ Discrete Event Systems
**Title:** D.E.S. model of a train station  
![MATLAB](https://img.shields.io/badge/MATLAB-Simulation-ED8B00?style=flat-square) ![CTHMC](https://img.shields.io/badge/Model-Continuous--Time%20Markov%20Chain-blue?style=flat-square)

> Implementation of a discrete event system simulating the operation of a train station with a single track. The model is formalized as an 11-state stochastic automaton, validated against analytical Continuous-Time Markov Chain limit probabilities under Poisson clocks, and evaluated under realistic event distributions with empirical convergence verified through the Law of Large Numbers.

🔗 **Links:** [📁 Browse Directory](./discrete-event-systems) • [📄 Final Report (PDF)](./discrete-event-systems/report.pdf)
