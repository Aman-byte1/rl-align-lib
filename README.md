# RL Align Lib

A comprehensive library for Reinforcement Learning from Human Feedback (RLHF) and direct preference alignment.

## Features
- **Online Policy Methods**: PPO, GRPO, TRPO, REINFORCE++
- **Offline Preference Methods**: DPO, IPO, KTO, ORPO
- **Specialized Reasoning Methods**: GDPO, GSPO, SDPO, RLVR

## Installation
```bash
pip install -e .
```

## Usage

### Direct Preference Optimization (DPO)
```python
from rl_align.agents.offline.dpo import DPOAgent
from rl_align.trainer import Trainer

# Initialize agent with model and reference model
agent = DPOAgent(config, model, ref_model)

# Train with your dataset
trainer = Trainer(agent, dataset, config=config)
trainer.train(epochs=3)
```

### Group Relative Policy Optimization (GRPO)
```python
from rl_align.agents.online.ppo_grpo import GRPOAgent

# Initialize agent (no critic model needed for GRPO)
agent = GRPOAgent(config, model)

# Train using relative rewards within groups
trainer = Trainer(agent, dataset, config=config)
trainer.train(epochs=3)
```

## Supported Algorithms
- **Online**: PPO, GRPO, GDPO, SDPO, TRPO, REINFORCE++
- **Offline**: DPO, IPO, KTO, ORPO
- **Reasoning**: RLVR (Rule-based Verifiers)
