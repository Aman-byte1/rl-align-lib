# RL Align Lib: Advanced Alignment for LLMs

A modular, extensible library for Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO). This library implements state-of-the-art alignment algorithms used in models like DeepSeek-V3 and Llama-3.

---

## 🚀 Supported Methods & Mathematical Foundation

### 1. Online Policy-Based Methods (Reward Modeling)

#### **PPO (Proximal Policy Optimization)**
The industry standard for RLHF. It uses an actor-critic architecture to optimize the policy while ensuring updates don't deviate too far from the previous policy.
- **Math**: Optimizes the clipped surrogate objective:
  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$
  where $r_t(\theta)$ is the probability ratio.

#### **GRPO (Group Relative Policy Optimization)**
Used in **DeepSeek-V3**, it eliminates the need for a critic model. It estimates the baseline by averaging rewards across a group of completions for the same prompt.
- **Math**: For a group of $G$ outputs $\{o_1, ..., o_G\}$, the advantage for $o_i$ is:
  $$\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$
  This significantly reduces memory overhead by removing the value network.

#### **TRPO & REINFORCE++**
- **TRPO**: Uses second-order optimization (Kullback-Leibler divergence constraint) to ensure stable updates.
- **REINFORCE++**: A modernized REINFORCE with variance reduction techniques and PPO-style clipping.

---

### 2. Direct Preference Optimization (DPO) & Variants

#### **DPO (Direct Preference Optimization)**
Bypasses the reward model by expressing the reward function in terms of the optimal policy.
- **Math**: The loss is defined as:
  $$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

#### **IPO (Identity Preference Optimization)**
A robust alternative to DPO that adds a root-mean-square error penalty to prevent the policy from collapsing.
- **Math**: $\mathcal{L}_{IPO} = (\text{log-ratio} - \frac{1}{2\beta})^2$.

#### **KTO (Kahneman-Tversky Optimization)**
Aligned with Prospect Theory, KTO doesn't require paired data (chosen/rejected). It works with binary "good" or "bad" signals.
- **Math**: Optimizes based on the utility function of gains and losses.

---

### 3. Specialized Reasoning & Sequence Methods

#### **GDPO (Group Reward-Decoupled Policy Optimization)**
Enhances diversity by decoupling different reward signals (e.g., accuracy vs. format) within the GRPO framework.

#### **SDPO (Step-wise Direct Preference Optimization)**
Optimizes sequences at the step level, crucial for multi-step reasoning tasks like math or coding.

#### **RLVR (Reinforcement Learning with Verifiable Rewards)**
Integrates rule-based verifiers (e.g., code execution results, math checkers) as the ground truth reward signal, ensuring the model optimizes for correctness rather than just looking correct.

---

## 🛠 Installation

```bash
git clone https://github.com/Aman-byte1/rl-align-lib.git
cd rl-align-lib
pip install -e .
```

---

## 📖 Quick Start

### Example: Training with GRPO (Group Relative)
GRPO is highly efficient as it doesn't require a critic model.

```python
from rl_align.agents.online.ppo_grpo import GRPOAgent
from rl_align.trainer import Trainer

# 1. Setup Configuration
config = {"beta": 0.1, "lr": 1e-5, "batch_size": 1}

# 2. Initialize Agent (Actor only)
agent = GRPOAgent(config, model)

# 3. Define Dataset with Grouped Samples
# Each sample should return a prompt and multiple completions with rewards
dataset = MyGroupedDataset()

# 4. Start Training
trainer = Trainer(agent, dataset, config=config)
trainer.train(epochs=3)
```

### Example: Training with DPO (Direct Preference)
```python
from rl_align.agents.offline.dpo import DPOAgent

# Requires a policy model and a frozen reference model
agent = DPOAgent(config, model, ref_model)

# Standard Trainer interface
trainer = Trainer(agent, dataset, config=config)
trainer.train(epochs=5)
```

---

## 📂 Project Structure
- `rl_align/agents/`: Implementation of all algorithms.
- `rl_align/trainer.py`: Unified training loop.
- `examples/`: Ready-to-run scripts for different methods.

---

## 🤝 Contributing
Contributions are welcome! Please see the implementation files in `rl_align/agents/` to add new variants.
