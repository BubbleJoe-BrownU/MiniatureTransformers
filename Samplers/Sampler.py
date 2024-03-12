import torch 

class Sampler:
    """
    An abstract sampler class that takes a logits tensor as input and returns a token tensor;
    """
    def __call__(self, logits: torch.Tensor):
        raise NotImplementedError(f"Attempting to instantiate an abstract class \"{self.__class__.__name__}\"")
    
