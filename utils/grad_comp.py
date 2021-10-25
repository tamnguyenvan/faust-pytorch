import torch

from .dvp import dvp
from .mult_left import mult_left
from .mult_right import mult_right


def norm(x, ord=2):
    if not torch.is_tensor(x):
        raise Exception(f'Input must be a Tensor, received: {type(x)}')
    
    if x.ndim == 1:
        raise Exception(f'Input dimension must be 0 or 2, received: {x.ndim}')
    
    if x.ndim == 0:
        return x
    else:
        return torch.linalg.norm(x, ord)


def grad_comp(L, S, R, X, lambda_, device):
    """Computation of the gradient and Lipschitz modulus

    [grad, LC] = grad_comp(L,S,R,X,lambda) computes the gradient grad of
    H(L,S,R,lambda) = || X - lambda*L*S*R || and its Lipschitz modulus LC.
    """
    grad_temp = lambda_ * mult_left(L, S)
    grad_temp = mult_right(grad_temp, R)
    grad_temp = grad_temp - X
    grad_temp = lambda_ * mult_right(grad_temp.T, L)
    grad_temp = mult_left(R, grad_temp)
    grad = grad_temp.T
    
    # Compute the Lipschitz constant
    LC = lambda_**2 * norm(dvp(R), 2)**2 * norm(dvp(L), 2)**2
    return grad, LC