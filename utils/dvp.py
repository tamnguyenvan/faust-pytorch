import torch


def dvp(D):
    """Development of the input factorized matrix.

    Ddvp = dvp(D) develops the cell-array of matrices D into the matrix Ddvp
    which is the product of the matrices contained in D:
    Ddvp = D{1}*D{2}*...*D{n}.
    """
    n_layers = len(D)
    if n_layers == 0:
        Ddvp = torch.tensor(1., dtype=torch.float32)
    else:
        Ddvp = D[0]
        for i in range(1, n_layers):
            if D[i].size(0) == Ddvp.size(1):
                Ddvp = Ddvp @ D[i]
    return Ddvp