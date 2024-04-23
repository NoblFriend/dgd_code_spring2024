from typing import Tuple, List
import torch
from utils.linalg.hosvd import sthosvd
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.decomposition import tensor_train

PYTORCH_ENABLE_MPS_FALLBACK=1

tl.set_backend('pytorch')

def tucker_decomposition(tensor, ranks):
    core, factors = tucker(tensor, rank=list(ranks))
    reconstructed_tensor = tl.tucker_to_tensor((core, factors))
    return reconstructed_tensor

def tucker_compression_op(target_dim=32):
    def compress(tensor):
        original_shape = tensor.shape
        new_shape = []

        for dim in original_shape:
            while dim > target_dim:
                if dim % 32 == 0:
                    new_shape.append(32)
                    dim //= 32
                else:
                    break
            new_shape.append(dim)

        significant_dims = [dim for dim in new_shape if dim > 1]
        if len(significant_dims) < 2:
            return tensor

        tensor = tensor.reshape(new_shape)

        ranks = [min(25, s) for s in new_shape]

        reconstructed_tensor = tucker_decomposition(tensor, ranks)
        reconstructed_tensor = reconstructed_tensor.reshape(original_shape)

        return reconstructed_tensor

    return compress

def tt_svd_decomposition(tensor, max_rank):
    tt_tensor = tensor_train(tensor, rank=max_rank)
    reconstructed_tensor = tl.tt_to_tensor(tt_tensor)
    return reconstructed_tensor

def tt_svd_compression_op(max_rank=25, target_dim=32):
    def compress(tensor):
        original_shape = tensor.shape
        new_shape = []

        for dim in original_shape:
            while dim > target_dim:
                if dim % target_dim == 0:
                    new_shape.append(target_dim)
                    dim //= target_dim
                else:
                    break
            new_shape.append(dim)

        significant_dims = [dim for dim in new_shape if dim > 1]
        if len(significant_dims) < 2:
            return tensor

        tensor = tensor.reshape(new_shape)
        ranks = [min(max_rank, s) for s in new_shape] + [1]
        ranks[0] = 1

        reconstructed_tensor = tt_svd_decomposition(tensor, ranks)
        reconstructed_tensor = reconstructed_tensor.reshape(original_shape)
        return reconstructed_tensor

    return compress

def top_k(k, tensor):
    new_tensor = tensor.clone()
    abs_tensor = torch.abs(tensor)
    kth_value = torch.kthvalue(abs_tensor.view(-1), k=abs_tensor.numel() - k + 1).values
    new_tensor[abs_tensor < kth_value] = 0
    return new_tensor

def top_k_op(k):
    return lambda t: top_k(k, t)

def hosvd(tensor, ranks):
    # Assuming the tensor is at least 2D and ranks are correctly set
    U_matrices = []
    core_tensor = tensor.clone()

    for mode in range(tensor.dim()):
        unfolded = tensor.unfold(mode, size=ranks[mode], step=ranks[mode])
        U, S, V = torch.svd(unfolded, some=True)
        U_reduced = U[:, :ranks[mode]]
        U_matrices.append(U_reduced)
        core_tensor = torch.tensordot(U_reduced.t(), unfolded, dims=([1], [mode]))

    reconstructed_tensor = core_tensor
    for i, U in enumerate(reversed(U_matrices)):
        reconstructed_tensor = torch.tensordot(reconstructed_tensor, U, dims=([0], [0]))
        permute_dims = list(range(1, reconstructed_tensor.dim())) + [0]
        reconstructed_tensor = reconstructed_tensor.permute(permute_dims)

    return reconstructed_tensor

def hosvd_compression_op(target_dim=16):
    def compress(tensor):
        original_shape = tensor.shape
        new_shape = []

        # Перебираем каждое измерение и делим его на 2 до достижения целевого размера
        for dim in original_shape:
            while dim > target_dim:
                if dim % 4 == 0:
                    new_shape.append(4)
                    dim //= 4
                else:
                    new_shape.append(dim)
                    break
            new_shape.append(dim)

        # Изменяем форму тензора
        tensor = tensor.reshape(new_shape)

        # Определение рангов для HOSVD
        ranks = [min(2, s) for s in new_shape]

        # Выполняем HOSVD
        reconstructed_tensor = hosvd(tensor, ranks)
        reconstructed_tensor = reconstructed_tensor.reshape(original_shape)
        return reconstructed_tensor

    return compress


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
        _, indices = torch.topk(abs_tensor.view(-1), min(self.k, abs_tensor.numel()), sorted=False)
        mask = torch.zeros_like(abs_tensor, dtype=torch.bool).view(-1)
        mask[indices] = True
        self.data_size = self.k
        return tensor * mask.view_as(tensor)
    

class HOSVD(CompOp):
    def __init__(self):
        pass

    def _make_dims(self, shape):
        new_shape = []
        for dim in shape:
            for div in [4, 5]:
                while dim % div == 0:
                    new_shape.append(div)
                    dim //= div
            if dim > 1:  
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