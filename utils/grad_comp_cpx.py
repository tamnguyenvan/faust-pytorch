import torch

from .dvp import dvp


def grad_comp_cpx(L, S, R, X, lambda_, device='cpu'):
    """Computation of the gradient and Lipschitz modulus for complex data

    [grad, LC] = grad_comp_cpx(L,S,R,X,lambda) computes the gradient grad of
    H(L,S,R,lambda) = 1/2|| X - lambda*L*S*R ||F and its Lipschitz modulus LC.
    """
    grad = lambda_*dvp(L, device).T*(lambda_*dvp(L, device)*S*dvp(R) - X)*dvp(R, device).T

    # Compute the Lipschitz constant
    LC = lambda_**2 * torch.linalg.norm(dvp(R, device), 2)**2 * torch.linalg.norm(dvp(L, device), 2)**2
    return grad, LC