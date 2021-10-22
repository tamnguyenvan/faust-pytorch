import torch


def normc(x):
    """Normalization of a matrix.
    y = normc(x) normalizes the columns of x, and puts the result in y.
    """
    n = 1 / (x**2).sum(dim=0).sqrt()
    y = x * n
    y[~torch.isfinite(y)] = 1.
    return y


def prox_sp(X, s):
    """Projection onto the set of sparse matrices of unit Frobenius norm.

    Xprox = prox_sp(X,s) projects the input matrix X onto the set of
    matrices which have at most s non-zero entries and unit Frobenius norm.
    Xprox is the projection of X onto this set.
    """
    Xabs = X.abs()
    Xprox = torch.zeros_like(X)
    N = X.numel()
    
    _, sorted_index = torch.sort(Xabs.view(-1), descending=True)
    max_index = sorted_index[:round(min(s, N))]
    Xprox_ = Xprox.view(-1)
    Xprox_[max_index] = X.view(-1)[max_index]
    Xprox = Xprox_.view_as(X)
    
    Xprox = Xprox / torch.linalg.norm(Xprox, 'fro')
    return Xprox


def prox_spcol(X, s):
    """Projection onto the set of matrices with sparse columns and 
    unit Frobenius norm.

    Xprox = prox_spcol(X,s) projects the input matrix X onto the set of
    matrices which have at most s non-zero entries per column and unit 
    Frobenius norm. Xprox is the projection of X onto this set.
    """
    Xabs = X.abs()
    Xprox = torch.zeros_like(X)

    _, sorted_index = torch.sort(Xabs, dim=0, descending=True)
    max_index = sorted_index[:round(s), :]
    col_index = torch.arange(X.size(1)).expand_as(max_index)

    Xprox[max_index, col_index] = X[max_index, col_index]
    Xprox = Xprox / torch.linalg.norm(Xprox, 'fro')
    return Xprox


def prox_normcol(X, s):
    """Projection onto the set of normalized matrices.

    Xprox = prox_normcol(X,s) projects the input matrix X onto the set of
    matrices which have all columns of norm s. Xprox is the projection of X
    onto this set.
    """
    Xprox = s * normc(X)
    return Xprox


def prox_normlin(X, s):
    """Projection onto the set of normalized matrices.

    Xprox = prox_normcol(X,s) projects the input matrix X onto the set of
    matrices which have all columns of norm s. Xprox is the projection of X
    onto this set.
    """
    Xprox = prox_normcol(X.T, s).T
    return Xprox


def prox_pos(X):
    """Projection onto the set of positive matrices of unit Frobenius norm.

    Xprox = prox_pos(X) projects the input matrix X onto the set of positive
    matrices which have unit Frobenius norm.
    Xprox is the projection of X onto this set.
    """
    Xprox = X * (X > 0)
    Xprox = Xprox / torch.linalg.norm(Xprox, 'fro')
    return Xprox


def prox_sp_pos(X, s):
    """Projection onto the set of sparse positive matrices of unit Frobenius 
    norm.

    Xprox = prox_sp_pos(X) projects the input matrix X onto the set of
    positive matrices which have at most s non-zero entries and unit
    Frobenius norm. Xprox is the projection of X onto this set.
    """
    Xprox = prox_sp(prox_pos(X), s)
    return Xprox


def prox_spcol(X, s):
    """Projection onto the set of matrices with sparse columns and 
    unit Frobenius norm.

    Xprox = prox_spcol(X,s) projects the input matrix X onto the set of
    matrices which have at most s non-zero entries per column and unit 
    Frobenius norm. Xprox is the projection of X onto this set.
    """
    Xabs = X.abs()
    Xprox = torch.zeros_like(X)

    _, sorted_index = torch.sort(Xabs, dim=0, descending=True)
    max_index = sorted_index[:round(s), :]
    col_index = torch.arange(X.size(1)).expand_as(max_index)
    Xprox[max_index, col_index] = X[max_index, col_index]
    Xprox = Xprox / torch.linalg.norm(Xprox, 'fro')
    return Xprox


def prox_splin(X, s):
    """Projection onto the set of matrices with sparse rows and unit Frobenius 
    norm.

    Xprox = prox_splin(X,s) projects the input matrix X onto the set of
    matrices which have at most s non-zero entries per row and unit 
    Frobenius norm. Xprox is the projection of X onto this set.
    """
    Xprox = prox_spcol(X.T, s).T
    return Xprox