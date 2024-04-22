from typing import Tuple, List
import torch


class CompOp:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, tensor):
        raise NotImplementedError
    

class TopK(CompOp):
    def __init__(self, k):
        self.k = k 

    def __call__(self, tensor):
        abs_tensor = torch.abs(tensor)
        _, indices = torch.topk(abs_tensor.view(-1), self.k, sorted=False)
        mask = torch.zeros_like(abs_tensor, dtype=torch.bool).view(-1)
        mask[indices] = True
        self.data_size = self.k
        return tensor * mask.view_as(tensor)