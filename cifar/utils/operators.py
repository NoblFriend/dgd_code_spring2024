from typing import Tuple, List
import torch
from utils.linalg.hosvd import sthosvd
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.decomposition import tensor_train
from math import log2, ceil

PYTORCH_ENABLE_MPS_FALLBACK=1

tl.set_backend('pytorch')

class CompOp:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, tensor):
        raise NotImplementedError
    
class ID(CompOp):
    def __init__(self):
       pass

    def __call__(self, tensor):
        self.data_size = tensor.numel()
        return tensor

class TopK(CompOp):
    def __init__(self, delta):
        if not (0 < delta <= 1):
            raise ValueError("comp factor must be in (0;1]")
        self.delta = delta

    def __call__(self, tensor):
        k = self.delta * abs_tensor.numel()
        abs_tensor = torch.abs(tensor)
        _, indices = torch.topk(abs_tensor.view(-1), k, sorted=False)
        mask = torch.zeros_like(abs_tensor, dtype=torch.bool).view(-1)
        mask[indices] = True
        self.data_size = k
        return tensor * mask.view_as(tensor)

def _make_dims(shape, target_dim):
    new_shape = []
    for dim in shape:
        while dim > target_dim:
            if dim % target_dim == 0:
                new_shape.append(target_dim)
                dim //= target_dim
            else:
                break
        new_shape.append(dim)
    return new_shape

    
class TTSVD(CompOp):
    def __init__(self, max_rank = 25, target_dim = 32):
        self.target_dim = target_dim
        self.max_rank = max_rank

    def __call__(self, tensor: torch.Tensor):
        shape = tensor.shape
        new_shape = _make_dims(shape, self.target_dim)

        significant_dims = [dim for dim in new_shape if dim > 1]
        if len(significant_dims) < 2:
            return tensor

        tensor = tensor.reshape(new_shape)
        ranks = [min(self.max_rank, s) for s in new_shape] + [1]
        ranks[0] = 1
        
        tt_tensor = tensor_train(tensor, rank=ranks)
        ranks = tt_tensor.rank
        dimensions = tt_tensor.shape
        self.data_size = sum(ranks[i-1] * dimensions[i-1] * ranks[i] for i in range(1, len(ranks)))

        reconstructed_tensor = tl.tt_to_tensor(tt_tensor)
        reconstructed_tensor = reconstructed_tensor.reshape(shape)
        return reconstructed_tensor
    

class TuckerOp(CompOp):
    def __init__(self, max_rank = 25, target_dim = 32):
        self.target_dim = target_dim
        self.max_rank = max_rank

    def __call__(self, tensor: torch.Tensor):
        shape = tensor.shape
        new_shape = _make_dims(shape, self.target_dim)

        significant_dims = [dim for dim in new_shape if dim > 1]
        if len(significant_dims) < 2:
            return tensor

        tensor = tensor.reshape(new_shape)
        ranks = [min(25, s) for s in new_shape]
        
        core, factors = tucker(tensor, rank=list(ranks))
        self.data_size = core.numel()
        for sv in factors:
            self.data_size += sv.numel()

        reconstructed_tensor = tl.tucker_to_tensor((core, factors))
        reconstructed_tensor = reconstructed_tensor.reshape(shape)
        return reconstructed_tensor
    
class HOSVD(CompOp):
    def __init__(self, max_rank = 25, target_dim = 32):
        self.target_dim = target_dim
        self.max_rank = max_rank

    def __call__(self, tensor: torch.Tensor):
        shape = tensor.shape
        new_shape = _make_dims(shape)

        significant_dims = [dim for dim in new_shape if dim > 1]
        if len(significant_dims) < 2:
            return tensor

        tensor = tensor.view(*new_shape)
        ranks = [min(25, s) for s in new_shape]

        core, svecs, _ = sthosvd(tensor, ranks)
        self.data_size = core.numel()
        for sv in svecs:
            self.data_size += sv.numel()

        reconstructed = tl.tucker_to_tensor((core, svecs)).reshape(shape)
        return reconstructed        

class NaturalDithering(CompOp):
    def __init__(self, n, p=2, b=2):
        self.n = n
        self.p = p
        self.b = b
        self.create_seq(n)
    
    def create_seq(self, n,  b=2):
        powers = torch.arange(-n, n, dtype=torch.float32)
        self.levels = torch.pow(b, -powers)
        self.delta = ceil(log2(2*n+1) + 1)/32

    def __call__(self, tensor: torch.Tensor):
        if self.levels.device != tensor.device:
            self.levels = self.levels.to(tensor.device)
        norm_p = tensor.norm("fro")
        dithered_tensor = torch.zeros_like(tensor)
        scaled_tensor = torch.abs(tensor) / norm_p
        
        # Expand levels to match the number of elements in scaled_tensor
        expanded_levels = self.levels.unsqueeze(0).expand(scaled_tensor.numel(), -1)
        
        # Calculate the index of the closest level for each element
        differences = torch.abs(expanded_levels - scaled_tensor.view(-1, 1))
        closest_levels_idx = torch.argmin(differences, dim=1)
        closest_levels = self.levels[closest_levels_idx]
        
        # Set the dithered values
        dithered_tensor.view(-1).copy_(torch.sign(tensor).view(-1) * closest_levels * norm_p)
        
        self.data_size = self.delta * tensor.numel()
        
        return dithered_tensor
