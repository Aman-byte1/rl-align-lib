# RL Align Lib: Advanced Alignment for LLMs

A modular, extensible library for Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO). This library implements state-of-the-art alignment algorithms used in models like DeepSeek-V3 and Llama-3.

---

## 🚀 Supported Methods & Mathematical Foundation

### 1. Online Policy-Based Methods (Reward Modeling)

#### **PPO (Proximal Policy Optimization)**
The industry standard for RLHF. It uses an actor-critic architecture to optimize the policy while ensuring updates don't deviate too far from the previous policy.
- **Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- **Math**: Optimizes the clipped surrogate objective:
  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$
  where $r_t(\theta)$ is the probability ratio.

#### **GRPO (Group Relative Policy Optimization)**
Used in **DeepSeek-V3** and **DeepSeek-R1**, it eliminates the need for a critic model. It estimates the baseline by averaging rewards across a group of completions for the same prompt.
- **Paper**: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)
- **Math**: For a group of $G$ outputs $\{o_1, ..., o_G\}$, the advantage for $o_i$ is:
  $$\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$
  This significantly reduces memory overhead by removing the value network.

#### **TRPO & REINFORCE++**
- **TRPO**: [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) (Schulman et al., 2015). Uses second-order optimization to ensure stable updates.
- **REINFORCE++**: A modernized REINFORCE with variance reduction techniques. See [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for practical implementations.

---

### 2. Direct Preference Optimization (DPO) & Variants

#### **DPO (Direct Preference Optimization)**
Bypasses the reward model by expressing the reward function in terms of the optimal policy.
- **Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
- **Math**: The loss is defined as:
  $$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

#### **IPO (Identity Preference Optimization)**
A robust alternative to DPO that adds a root-mean-square error penalty to prevent the policy from collapsing.
- **Paper**: [A General Theoretical Paradigm for Deterministic and Stochastic Optimization](https://arxiv.org/abs/2310.12036) (Azar et al., 2023)
- **Math**: $\mathcal{L}_{IPO} = (\text{log-ratio} - \frac{1}{2\beta})^2$.

#### **KTO (Kahneman-Tversky Optimization)**
Aligned with Prospect Theory, KTO doesn't require paired data (chosen/rejected). It works with binary "good" or "bad" signals.
- **Paper**: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) (Ethayarajh et al., 2024)
- **Math**: Optimizes based on the utility function of gains and losses.

#### **ORPO (Odds Ratio Preference Optimization)**
Combines the supervised fine-tuning (SFT) and alignment stages into one, using an odds ratio penalty to discourage rejected responses.
- **Paper**: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) (Hong et al., 2024)

---

### 3. Specialized Reasoning & Sequence Methods

#### **GDPO (Group Reward-Decoupled Policy Optimization)**
Enhances diversity by decoupling different reward signals. See [DeepSeek-V3](https://arxiv.org/abs/2412.19437) for related group-based optimization concepts.

#### **SDPO (Step-wise Direct Preference Optimization)**
Optimizes sequences at the step level.
- **Paper**: [Step-wise Direct Preference Optimization for LLM Alignment](https://arxiv.org/abs/2406.11831) (2024)

#### **RLVR (Reinforcement Learning with Verifiable Rewards)**
Integrates rule-based verifiers as the ground truth reward signal.
- **Reference**: Popularized by OpenAI's and DeepSeek's reasoning model training (e.g., [DeepSeek-R1](https://arxiv.org/abs/2501.12948)).

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
