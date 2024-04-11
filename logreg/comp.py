import numpy as np
from scipy.linalg import svd


def unfold(tensor, mode):
    sz = range(len(tensor.shape))
    new_tensor = np.moveaxis(tensor, sz, np.roll(sz, mode))
    return np.reshape(new_tensor, (tensor.shape[mode], -1))

def hosvd_approx(tensor, ranks):
    U_matrices = []

    core_tensor = tensor
    for mode in range(len(tensor.shape)):
        U, _, _ = svd(unfold(tensor, mode), full_matrices=False)
        U_full = np.zeros_like(U)
        min_dim = min(U.shape[1], ranks[mode])
        U_full[:, :min_dim] = U[:, :min_dim]

        U_matrices.append(U_full)

        core_tensor = np.moveaxis(
            np.tensordot(core_tensor, U_full.T, axes=[mode, 1]), -1, mode
        )
    
    tensor = core_tensor

    for mode, U in enumerate(U_matrices):
        tensor = np.moveaxis(np.tensordot(tensor, U, axes=[mode, 1]), -1, mode)

    return tensor

def hosvd_approximation_low_rank_compression(arr, rank):
    #print(arr)
    X_matrix = arr.reshape(7, 2, 2, 2, 2)
    x_compressed = hosvd_approx(X_matrix, [1, 1, 1, 1, 1])
    x_compressed = x_compressed.flatten()
    #print(x_compressed)
    return x_compressed

def hosvd_approximation_low_rank_compression_op(rank=1):
    return lambda arr: hosvd_approximation_low_rank_compression(arr, rank)

def top_k(k, arr):
    new_arr = arr.copy()
    abs_arr = np.abs(arr)
    new_arr[abs_arr < np.sort(abs_arr)[-k]] = 0
    return new_arr

def _generate_alpha_sequence(n, start=0.01, factor=2):
    return [start * (factor ** k) for k in range(n)]

def general_biased_rounding(x, n, start=0.01, factor=2):
    alpha_sequence = _generate_alpha_sequence(n, start, factor)
    C_x = np.zeros_like(x)
    for i, x_i in enumerate(x):
        # Поиск ближайшего элемента t из alpha_sequence к |x_i|
        t_min = min(alpha_sequence, key=lambda t: abs(t - abs(x_i)))
        C_x[i] = np.sign(x_i) * t_min
    return C_x

def natural_dithering(x, n, p=2, b=2):
    norm_p = np.linalg.norm(x, p)
    dithered_x = np.zeros_like(x)

    for i, xi in enumerate(x):
        levels = [b**(-l) for l in range(n)]
        closest_levels = min(levels, key=lambda l: abs(l - abs(xi) / norm_p))
        dithered_x[i] = np.sign(xi) * closest_levels * norm_p
    return dithered_x

def svd_approximation_low_rank_compression(x, rank=2):
    X_matrix = x.reshape(14, 8)
    U, s, Vt = np.linalg.svd(X_matrix, full_matrices=False)

    S = np.diag(s[:rank])
    U_low_rank = U[:, :rank]
    Vt_low_rank = Vt[:rank, :]

    X_approx = np.dot(U_low_rank, np.dot(S, Vt_low_rank))
    x_compressed = X_approx.reshape(x.shape)
    return x_compressed

def top_k_op(k):
    return lambda arr: top_k(k, arr)

def general_biased_rounding_op(n, start=0.01, factor=2):
    return lambda arr: general_biased_rounding(arr, n, start, factor)

def natural_dithering_op(n, p=2, b=2):
    return lambda arr: natural_dithering(arr, n, p, b)

def svd_approximation_low_rank_compression_op(rank=2):
    return lambda arr: svd_approximation_low_rank_compression(arr, rank)

def identity_op():
    return lambda arr: arr