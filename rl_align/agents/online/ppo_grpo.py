import torch
import torch.nn.functional as F
from ..base import BaseAgent
from transformers import PreTrainedModel
from typing import Dict, Any, List

class GRPOAgent(BaseAgent):
    """
    Group Relative Policy Optimization (GRPO) for LLMs.
    As described in the DeepSeek-V3 and R1 reports.
    """
    def __init__(self, config: Dict[str, Any], model: PreTrainedModel):
        super().__init__(config)
        self.model = model
        self.beta = config.get("beta", 0.1)

    def get_logps(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        mask = shift_labels != -100
        return (per_token_logps * mask).sum(-1)

    def compute_loss(self, batch):
        """
        Expects batch with:
        - input_ids: [batch_size * group_size, seq_len]
        - attention_mask: [batch_size * group_size, seq_len]
        - rewards: [batch_size * group_size]
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        rewards = batch["rewards"]
        group_size = self.config.get("group_size", 8)
        
        # Reshape rewards to calculate relative advantages per group
        # rewards: [B * G] -> [B, G]
        b_g = rewards.shape[0]
        b = b_g // group_size
        rewards_reshaped = rewards.view(b, group_size)
        
        mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
        std_rewards = rewards_reshaped.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_reshaped - mean_rewards) / std_rewards
        advantages = advantages.view(-1) # back to [B * G]

        # Forward pass
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        logps = self.get_logps(logits, input_ids)
        
        # KL Divergence with reference (optional but recommended)
        # For simplicity, we assume the user might provide ref_logps in batch
        if "ref_logps" in batch:
            ref_logps = batch["ref_logps"]
            kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
            loss = -(advantages * torch.exp(logps - logps.detach())) + self.beta * kl
        else:
            loss = -(advantages * logps)
            
        return loss.mean()
