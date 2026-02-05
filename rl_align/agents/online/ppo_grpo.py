import torch
import torch.nn.functional as F
from ..base import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, config, actor, critic):
        super().__init__(config)
        self.actor = actor
        self.critic = critic
        self.clip_range = config.get("clip_range", 0.2)
        self.ent_coef = config.get("ent_coef", 0.01)

    def compute_loss(self, batch):
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(obs)
        value_loss = F.mse_loss(values, returns)

        loss = policy_loss + self.config.get("vf_coef", 0.5) * value_loss - self.ent_coef * entropy
        return loss

class GRPOAgent(BaseAgent):
    """Group Relative Policy Optimization (GRPO)"""
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model
        self.beta = config.get("beta", 0.1)

    def compute_loss(self, batch):
        # batch contains a group of completions for the same prompt
        prompts = batch["prompts"]
        completions = batch["completions"]
        rewards = batch["rewards"] # list of rewards for the group
        
        # Calculate relative rewards within the group
        mean_reward = torch.mean(rewards)
        std_reward = torch.std(rewards) + 1e-8
        advantages = (rewards - mean_reward) / std_reward

        # Policy loss using relative advantages
        logits = self.model(prompts, completions)
        # Simplified loss logic for GRPO
        loss = -(advantages * logits.log_prob).mean()
        return loss
