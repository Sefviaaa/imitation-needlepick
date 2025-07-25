# Imitation Learning for Needle Pick Task using SurRoL

This repository contains the source code and experimental results of my project on **Imitation Learning** applied to the **Needle Pick** surgical task using the **SurRoL** (Surgical Robot Learning) environment.

## Credits

This project builds upon and gives credit to [SurRoL](https://github.com/med-air/SurRoL):  
**"[IROS'21] SurRoL: An Open-source Reinforcement Learning Centered and dVRK Compatible Platform for Surgical Robot Learning"**  
We acknowledge the contributions of the SurRoL team and their open-source RL platform for surgical robot learning.

---

## 📌 Overview

The goal of this project is to evaluate and compare multiple imitation learning models for automating the needle pick task in a simulated surgical environment. The project involves expert demonstration collection, model training, hyperparameter tuning, and final evaluation using the **Success Rate and Episode Return** metric.

---

## 🧪 Methodology

### 🗂 Dataset Collection

- Expert demonstrations were collected from the SurRoL environment using the built-in oracle controller:
  - `100-episodes/`
  - `2000-episodes/`

---

### 🔧 Model Training

Six models were trained and evaluated:

 1️⃣ MLP | Behavior Cloning using Multilayer Perceptron 
 2️⃣ LSTM | Behavior Cloning using Long Short-Term Memory 
 3️⃣ DAgger MLP | DAgger algorithm using MLP 
 4️⃣ DAgger MLP (Tuned) | DAgger MLP with tuning 
 5️⃣ DAgger LSTM | DAgger algorithm using LSTM 
 6️⃣ DAgger LSTM (Tuned) | DAgger LSTM with tuning 

Each model undergoes **grid search** over hyperparameters:
- Learning rate
- Hidden size
- Batch size
- Epochs
- Number of layers

The best configuration is selected based on **Success Rate and Episode Return** on the Needle Pick environment.



### 🔁 Final Evaluation

- After hyperparameter tuning, all six models were **re-trained using the best configuration** with **5 different random seeds**.
- Models were then evaluated on the SurRoL environment to compare performance.

---


## 📊 Evaluation Metric

The primary evaluation metric is:

**✅ Success Rate** =  
Number of successful episodes / Total evaluation episodes × 100%

---

## 💻 Environment

- Python 3.8+
- SurRoL (Surgical Robot Learning)
- PyTorch
- Gym / Gymnasium
- Stable-Baselines3
- Imitation Learning Library

All dependencies and setup instructions are in the `setup/` folder.

---

## 📈 Results Summary

Results and plots for each experiment (including training curves, success rates, and comparisons across seeds) are stored in the `training/` folder.

---

## Installation
This project is built with Python 3.9, Gym 0.26.2, PyBullet, and Stable-Baselines3 2.2.1. 

# Clone Repository
To get started, clone this repository:

```bash
git clone https://github.com/Sefviaaa/imitation-needlepick.git
cd imitation-needlepick
```

# Prepare Environment
Using the surrol_env.yml in the setup folder, create a conda environment and install dependencies as follows:
```bash
conda env create -f surrol_env.yml
conda activate surrol_env
```

Then, install SurRoL
```bash
cd SurRoL
pip install -e .
```







