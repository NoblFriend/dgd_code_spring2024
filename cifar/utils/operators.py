from typing import Tuple, List
import torch
from utils.linalg.hosvd import sthosvd
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.decomposition import tensor_train

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
    def __init__(self, k):
        self.k = k 

    def __call__(self, tensor):
        abs_tensor = torch.abs(tensor)
        _, indices = torch.topk(abs_tensor.view(-1), min(self.k, abs_tensor.numel()), sorted=False)
        mask = torch.zeros_like(abs_tensor, dtype=torch.bool).view(-1)
        mask[indices] = True
        self.data_size = self.k
        return tensor * mask.view_as(tensor)
    

class HOSVD(CompOp):
    def __init__(self):
        pass

   def _make_dims(self, shape, target_dim):
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


    def __call__(self, tensor: torch.Tensor):
        shape = tensor.shape
        new_shape = self._make_dims(shape)

        tensor = tensor.view(*new_shape)

        core, svecs, _ = sthosvd(tensor, [3] * len(new_shape))

        reconstructed = tl.tucker_to_tensor((core, svecs)).reshape(shape)

        self.data_size = core.numel()
        for sv in svecs:
            self.data_size += sv.numel()

        return reconstructed
    
class TTSVD(CompOp):
    def __init__(self, max_rank = 25, target_dim = 32):
        self.target_dim = target_dim
        self.max_rank = max_rank

    def _make_dims(self, shape, target_dim):
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


    def __call__(self, tensor: torch.Tensor):
        shape = tensor.shape
        new_shape = self._make_dims(shape, self.target_dim)

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

    def _make_dims(self, shape, target_dim):
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


    def __call__(self, tensor: torch.Tensor):
        shape = tensor.shape
        new_shape = self._make_dims(shape, self.target_dim)

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