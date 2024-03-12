import torch
import torch.nn as nn
from torch.distributions import Categorical
import Sampler


class TemperatureSampler(Sampler):
    """
    A TemperatureSampler simply divide every logit by the temperature 
    and feed the output to softmax
    """
    def __init__(self, temperature: float=1.0):
        self.t = temperature

    def __call__(self, logits: torch.Tensor):
        dist = Categorical(logits=logits/self.t)
        return dist.sample()