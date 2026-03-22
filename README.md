# Physics-Embedded Graph Neural Operator for Interaction-Controlled Colloidal Aggregation

**Published in [Water Research](https://www.sciencedirect.com/journal/water-research)** (2026)
[[paper]](https://www.sciencedirect.com/science/article/pii/S0043135426004550)

### ✍️ Authors

**Yongjoon Choe**<sup>a,\*</sup>, **Sungwon Kim**<sup>b,\*</sup>, **Susan E. Burns**<sup>a,✉</sup>, **Chanyoung Park**<sup>c,✉</sup>

<sup>a</sup> School of Civil and Environmental Engineering, Georgia Institute of Technology, 790 Atlantic Drive, N.W., Atlanta, GA, 30332-0355, Georgia  
<sup>b</sup> Graduate School of Data Science, KAIST, Daejeon, Republic of Korea  
<sup>c</sup> Department of Industrial and Systems Engineering, KAIST, Daejeon, Republic of Korea  

<sup>*</sup> These authors contributed equally to this work
<sup>✉</sup> Corresponding authors

---
This repository contains the official PyTorch implementation of the paper **"Physics-Embedded Graph Neural Operator for Interaction-Controlled Colloidal Aggregation"**.

## 📖 Overview

**GNOPB** (Graph Neural Operator for Population Balance) is a deep learning surrogate model designed to predict colloidal aggregation dynamics. Unlike standard black-box models or physics-informed neural networks (PINNs) that rely on heavy loss-term regularization, GNOPB embeds **Brownian perikinetic particle collision physics directly into the graph architecture**.

**Key Features:**
* **Physics-Embedded Graph:** Particle size classes are represented as nodes, with Brownian collision kernels encoded into edge features to strictly follow transport physics.
* **Generalization:** Achieves $R^2 > 0.99$ across diverse electrochemical conditions (Ionic Strength and Zeta Potential) and initial concentrations.
* **Efficiency:** Offers superior computational speed compared to traditional PBE solvers and physics-informed loss approaches.

---

## ⚙️ Environment Setup

This project uses **Conda** for environment management.

### 1. Configure Environment Path
Before creating the environment, you must update the `prefix` path in the `env.yaml` file to match your local Anaconda installation directory.

1.  Open `env.yaml`.
2.  Locate the line: `prefix: /path/to/your/anaconda3/envs/pbe`.
3.  **Action:** Change `/path/to/your/anaconda3/envs/pbe` to your actual local path.

### 2. Install Dependencies
Run the following commands to create and activate the environment:

```bash
# Create the environment from the file
conda env create --file env.yaml

# Activate the environment
conda activate pbe
```

---

## 📊 Experiment Tracking (WandB)

We use **Weights & Biases (WandB)** for real-time experiment tracking and result visualization.

1.  **Sign Up:** Create an account at [wandb.ai](https://wandb.ai/home).
2.  **Login:** Run the following command in your terminal:
    ```bash
    wandb login
    ```
3.  **Authorize:** Copy your API key from [wandb.ai/authorize](https://wandb.ai/authorize) and paste it into the terminal when prompted.

Once authenticated, training metrics and result plots will be automatically logged to your WandB dashboard.

---
## 📂 Data Setup

To comply with GitHub's file size limits, the `data` folder has been compressed and split into multiple parts (`data.tar.gzaa`, `data.tar.gzab`).

Before running any scripts, you must **recombine and extract** these files. Run the following command in the root directory:

```bash
cat data.tar.* | tar xvzf -
```
---

## 🚀 Reproducing Results

The following shell scripts reproduce the key results presented in **Table 1** and **Table 2** of the paper.

### 1. Generalization on Unseen Parameters (Table 1)  
Evaluates the model's performance on electrochemical conditions (IS, $\zeta$, $N_0$, $r_{init}$) not seen during training.

```bash
cd sh
sh table1_unseen_params.sh
```

### 2. Temporal Extrapolation (Table 1)  
Evaluates the model's ability to recursively predict future time steps beyond the training window.  
```bash
cd sh
sh table1_unseen_time.sh
```

### 3. Comparison with Physics-Informed Loss (Table 2)
Compares the proposed architecture (GNOPB) against Physics-Informed Neural Networks (MLP+PINN) regarding accuracy and computational cost.
```bash
cd sh
sh table2_phyloss.sh
```

---

## 📝 Citation

This paper has been published in **Water Research**. If you find this work useful, please cite: 

```bibtex
@article{CHOE2026125773,
title = {Physics-Embedded Graph Neural Operator for Interaction-Controlled Colloidal Aggregation},
journal = {Water Research},
pages = {125773},
year = {2026},
issn = {0043-1354},
doi = {https://doi.org/10.1016/j.watres.2026.125773},
url = {https://www.sciencedirect.com/science/article/pii/S0043135426004550},
author = {Yongjoon Choe and Sungwon Kim and Susan E. Burns and Chanyoung Park},
}
```
