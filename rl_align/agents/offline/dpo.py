import torch
import torch.nn.functional as F
from ..base import BaseAgent

class DPOAgent(BaseAgent):
    def __init__(self, config, model, ref_model):
        super().__init__(config)
        self.model = model
        self.ref_model = ref_model
        self.beta = config.get("beta", 0.1)
        self.label_smoothing = config.get("label_smoothing", 0.0)
        self.loss_type = config.get("loss_type", "sigmoid") # sigmoid, ipo, kto

    def get_logps(self, logits, labels):
        # Helper to compute log probabilities
        logps = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(logps, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return per_token_logps.sum(-1)

    def compute_loss(self, batch):
        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        chosen_labels = batch["chosen_labels"]
        rejected_labels = batch["rejected_labels"]

        policy_chosen_logits = self.model(chosen_input_ids).logits
        policy_rejected_logits = self.model(rejected_input_ids).logits
        
        with torch.no_grad():
            ref_chosen_logits = self.ref_model(chosen_input_ids).logits
            ref_rejected_logits = self.ref_model(rejected_input_ids).logits

        policy_chosen_logps = self.get_logps(policy_chosen_logits, chosen_labels)
        policy_rejected_logps = self.get_logps(policy_rejected_logits, rejected_labels)
        ref_chosen_logps = self.get_logps(ref_chosen_logits, chosen_labels)
        ref_rejected_logps = self.get_logps(ref_rejected_logits, rejected_labels)

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - \
                     F.logsigmoid(-self.beta * logits) * self.label_smoothing
        elif self.loss_type == "ipo":
            losses = (logits - 1/(2 * self.beta)) ** 2
        elif self.loss_type == "kto":
            # Simplified KTO logic
            chosen_losses = 1 - torch.sigmoid(self.beta * (policy_chosen_logps - ref_chosen_logps))
            rejected_losses = 1 - torch.sigmoid(self.beta * (ref_rejected_logps - policy_rejected_logps))
            losses = torch.cat([chosen_losses, rejected_losses])
        
        return losses.mean()

    def update(self, batch):
        loss = self.compute_loss(batch)
        # Optimization logic would go here in a full trainer
        return loss.item()

class ORPOAgent(DPOAgent):
    def compute_loss(self, batch):
        # ORPO combines SFT loss with an odds ratio penalty
        # Implementation of ORPO specific loss logic
        pass
