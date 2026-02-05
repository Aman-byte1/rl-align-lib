from rl_align.agents.online.ppo_grpo import GRPOAgent
from rl_align.trainer import Trainer
import torch
from transformers import AutoModelForCausalLM

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Setup config
config = {
    "beta": 0.1,
    "lr": 1e-5,
    "batch_size": 1,
}

# 3. Initialize Agent
agent = GRPOAgent(config, model)

# 4. Dummy Dataset with Grouped Rewards
class GroupedDataset(torch.utils.data.Dataset):
    def __len__(self): return 10
    def __getitem__(self, idx):
        # In GRPO, we compare a group of outputs for the same prompt
        return {
            "prompts": torch.randint(0, 100, (5,)),
            "completions": torch.randint(0, 100, (8, 5,)), # 8 completions
            "rewards": torch.randn(8,), # Rewards for each completion
        }

dataset = GroupedDataset()

# 5. Train
trainer = Trainer(agent, dataset, config=config)
trainer.train(epochs=1)
