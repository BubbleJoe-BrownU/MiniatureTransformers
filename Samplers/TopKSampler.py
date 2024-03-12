import torch
import torch.nn.functional as F
import Sampler

class TopKSampler(Sampler):
    def __init__(self, k:int, sampler):
        self.k = k
        self.sampler = sampler

    def __call__(self, logits: torch.Tensor):
        k = min(self.k, logits.shape[-1])
        # only keep top k logits, filling other entries with -inf
        probs = logits.new_ones(logits.shape) * float("-inf")
        top_k_values, indices = torch.topk(logits, k, dim=-1)
        probs.scatter_(-1, indices, top_k_values)
        return self.sampler(probs)
        
        