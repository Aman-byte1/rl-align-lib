from rl_align.agents.offline.dpo import DPOAgent
from rl_align.trainer import Trainer
import torch
from transformers import AutoModelForCausalLM

# 1. Load models
model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Setup config
config = {
    "beta": 0.1,
    "lr": 5e-5,
    "batch_size": 2,
    "loss_type": "sigmoid"
}

# 3. Initialize Agent
agent = DPOAgent(config, model, ref_model)

# 4. Dummy Dataset (Replace with your actual dataset)
class DummyDataset(torch.utils.data.Dataset):
    def __len__(self): return 10
    def __getitem__(self, idx):
        return {
            "chosen_input_ids": torch.randint(0, 100, (10,)),
            "rejected_input_ids": torch.randint(0, 100, (10,)),
            "chosen_labels": torch.randint(0, 100, (10,)),
            "rejected_labels": torch.randint(0, 100, (10,)),
        }

dataset = DummyDataset()

# 5. Train
trainer = Trainer(agent, dataset, config=config)
trainer.train(epochs=1)
