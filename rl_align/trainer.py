import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        agent,
        dataset,
        reward_model=None,
        config=None,
    ):
        self.agent = agent
        self.dataset = dataset
        self.reward_model = reward_model
        self.config = config or {}
        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(), 
            lr=self.config.get("lr", 1e-5)
        )

    def train(self, epochs=1):
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.config.get("batch_size", 4),
            shuffle=True
        )
        
        self.agent.train()
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                self.optimizer.zero_grad()
                
                # If it's an online method and we have a reward model
                if self.reward_model and hasattr(self.agent, 'generate'):
                    # 1. Generate completions
                    # 2. Get rewards from self.reward_model
                    # 3. Add to batch
                    pass

                loss = self.agent.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                
                pbar.set_postfix({"loss": loss.item()})

    def save(self, path):
        torch.save(self.agent.state_dict(), path)
