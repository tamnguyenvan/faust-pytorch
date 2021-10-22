import torch


def nnzero_count(D):
    """Compute the number of non-zero values of a FAuST structure D.
    total_l_0 = nnzero_count(D)
    """
    total_l_0 = 0
    n_layers = len(D)
    for i in range(n_layers):
        total_l_0 += torch.count_nonzero(D[i])
    return total_l_0