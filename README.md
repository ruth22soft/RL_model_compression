# RL_model_compression
# Model Compression for Deep Reinforcement Learning (DQN)

This project investigates the impact of **model compression techniques** on a **Deep Q-Network (DQN)** trained for the **LunarLander** reinforcement learning environment. The study evaluates how pruning and quantization affect **model size, inference time, and policy performance**, highlighting the trade-offs involved when deploying RL models in resource-constrained settings.

---

## 📌 Project Objectives

- Train a baseline DQN agent for the LunarLander environment
- Apply model compression techniques:
  - Unstructured weight pruning
  - Post-training dynamic quantization
  - Combined pruning and quantization
- Measure and compare:
  - Average reward
  - Policy accuracy
  - Inference time per step
  - Model size
- Analyze performance–efficiency trade-offs

---

## 🧠 Environment and Task

- **Environment:** `LunarLander-v3`
- **Observation Space:** 8-dimensional continuous state vector
- **Action Space:** 4 discrete actions
- **Algorithm:** Deep Q-Network (DQN)

---

## 🏗️ Model Architecture

The DQN consists of:
- Input layer: state dimension (8)
- Hidden layers:
  - Fully connected layer (256 units, ReLU)
  - Fully connected layer (256 units, ReLU)
- Output layer: action-value estimates (Q-values)

---

## ⚙️ Compression Techniques

### 1. Pruning
- Removes 30% of low-magnitude weights from linear layers
- Goal: reduce computational complexity
- Trade-off: may degrade policy stability in RL settings

### 2. Quantization
- Dynamic post-training quantization (INT8)
- Converts 32-bit floating-point weights to 8-bit integers
- Significantly reduces model size with minimal performance loss

### 3. Pruning + Quantization
- Applies both techniques sequentially
- Demonstrates the effect of aggressive compression

---

## 📊 Experimental Results

| Model | Avg Reward | Accuracy | Inference Time (ms) | Model Size (MB) |
|------|-----------|----------|--------------------|-----------------|
| Base DQN | 100.33 | 42% | 0.1491 | 0.2666 |
| Pruned (30%) | 67.94 | 42% | 0.1287 | 0.2666 |
| Quantized | **109.70** | **50%** | 0.3304 | **0.0721** |
| Pruned + Quantized | 25.41 | 28% | 0.3108 | 0.0721 |

---

## 🔍 Key Findings

- Quantization provided the **best trade-off**, reducing model size by ~73% while improving reward and accuracy
- Pruning alone slightly reduced inference time but degraded policy performance
- Combining pruning and quantization led to excessive information loss
- RL models are particularly sensitive to aggressive compression

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install gymnasium[box2d] torch numpy
