import torch
import torch.nn.functional as F
from ..base import BaseAgent
from transformers import PreTrainedModel
from typing import Optional, Union, Dict, Any

class DPOAgent(BaseAgent):
    def __init__(
        self, 
        config: Dict[str, Any], 
        model: PreTrainedModel, 
        ref_model: Optional[PreTrainedModel] = None
    ):
        super().__init__(config)
        self.model = model
        self.ref_model = ref_model
        self.beta = config.get("beta", 0.1)
        self.label_smoothing = config.get("label_smoothing", 0.0)
        self.loss_type = config.get("loss_type", "sigmoid")

    def get_logps(self, logits, labels):
        """Compute log probabilities for the given labels."""
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens if necessary (assuming 0 or -100 as ignore index)
        mask = shift_labels != -100
        return (per_token_logps * mask).sum(-1)

    def compute_loss(self, batch):
        # Support standard Hugging Face batch format
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch.get("chosen_attention_mask")
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch.get("rejected_attention_mask")
        
        # Policy forward pass
        policy_chosen_logits = self.model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
        policy_rejected_logits = self.model(rejected_input_ids, attention_mask=rejected_attention_mask).logits
        
        # Reference forward pass
        with torch.no_grad():
            if self.ref_model is None:
                # Use policy model with disabled adapters if using PEFT
                with self.model.disable_adapter():
                    ref_chosen_logits = self.model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
                    ref_rejected_logits = self.model(rejected_input_ids, attention_mask=rejected_attention_mask).logits
            else:
                ref_chosen_logits = self.ref_model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
                ref_rejected_logits = self.ref_model(rejected_input_ids, attention_mask=rejected_attention_mask).logits

        policy_chosen_logps = self.get_logps(policy_chosen_logits, chosen_input_ids)
        policy_rejected_logps = self.get_logps(policy_rejected_logits, rejected_input_ids)
        ref_chosen_logps = self.get_logps(ref_chosen_logits, chosen_input_ids)
        ref_rejected_logps = self.get_logps(ref_rejected_logits, rejected_input_ids)

        logits = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - \
                     F.logsigmoid(-self.beta * logits) * self.label_smoothing
        elif self.loss_type == "ipo":
            losses = (logits - 1/(2 * self.beta)) ** 2
        
        return losses.mean()
