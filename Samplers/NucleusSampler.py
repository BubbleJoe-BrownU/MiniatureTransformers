import torch
import torch.nn as nn
import Sampler

class NucleusSampler(Sampler):
    def __init__(self, p: float, sampler: Sampler):
        self.p = p
        self.sampler = sampler

    def __call__(self, logits: torch.Tensor):
        probs = nn.functional.softmax(logits, dim=-1)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < self.p
        # prepend ones s.t. we have at least one token with cumulative probability less than p
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1, )), nucleus[..., :-1]], dim=-1)
        # get the log probabilities and mask out the non-nucleus
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float("-inf")
        sampled_sorted_index = self.sampler(sorted_log_probs)
        return indices.gather(-1, sampled_sorted_index.unsqueeze(-1))




