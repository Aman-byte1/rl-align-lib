import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAgent(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def update(self, batch):
        """Update the policy based on a batch of data."""
        pass

    @abstractmethod
    def compute_loss(self, batch):
        """Compute the loss for the algorithm."""
        pass
