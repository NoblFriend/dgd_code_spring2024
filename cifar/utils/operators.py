import torch

def top_k(k, tensor):
    new_tensor = tensor.clone()
    abs_tensor = torch.abs(tensor)
    kth_value = torch.kthvalue(abs_tensor.view(-1), k=abs_tensor.numel() - k + 1).values
    new_tensor[abs_tensor < kth_value] = 0
    return new_tensor

def top_k_op(k):
    return lambda t: top_k(k, t)

def hosvd(tensor, ranks):
    """ Perform higher-order singular value decomposition (HOSVD) on a tensor. """
    U_matrices = []
    core_tensor = tensor.clone()

    for mode in range(tensor.dim()):
        # Unfold the tensor along the specified mode
        unfolded = tensor.permute(mode, *[i for i in range(tensor.dim()) if i != mode]).reshape(tensor.shape[mode], -1)
        # Compute the SVD of the unfolded matrix
        U, S, V = torch.svd(unfolded, some=True)  # Use some=True to compute the reduced SVD
        # Reduce U to the specified rank
        U_reduced = U[:, :ranks[mode]]
        U_matrices.append(U_reduced)
        # Reduce the core tensor by projecting onto the reduced U matrix
        core_tensor = (U_reduced.t() @ unfolded).reshape((ranks[mode],) + tensor.shape[1:mode+1] + tensor.shape[mode+1:])
        tensor = core_tensor.permute(1, *range(2, tensor.dim()), 0)

    # Reconstruct the tensor from the core tensor and the U matrices
    for i, U in enumerate(U_matrices):
        core_tensor = core_tensor.permute(-1, *range(core_tensor.dim() - 1))
        core_tensor = U @ core_tensor.reshape(U.shape[1], -1)
        core_tensor = core_tensor.reshape(tensor.shape)

    return core_tensor

def hosvd_compression_op():
    """ Return a compression operator for HOSVD with automatic rank determination based on tensor dimensions. """
    def compress(tensor):
        # Set ranks to 2 for each dimension, or to the dimension size if it is less than 2
        ranks = [min(2, s) for s in tensor.shape]
        return hosvd(tensor, ranks)
    return compress
