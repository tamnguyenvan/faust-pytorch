import math
import torch

from proxs import prox_sp, prox_normcol, prox_sp_pos, prox_spcol, prox_splin
from utils import mult_right, dvp, grad_comp, grad_comp_cpx


def palm4msa(params):
    """Factorization of a data matrix into multiple factors using PALM.
    lambda, facts = palm4MSA(params) runs the PALM algorithm on the
    specified set of signals (Algorithm 2 of [1]), returning the factors in
    "facts" and the multiplicative scalar in "lambda".

    Required fields in PARAMS:
    --------------------------

    'data' - Training data.
        A matrix containing the training signals as its columns.

    'nfacts' - Number of factors.
        Specifies the desired number of factors.

    'cons' - Constraint sets.
        Specifies the constraint sets in which each factor should lie. It
        should be a cell-array of size 1*nfacts, where the jth column sets
        the constraints for the jth factor (starting from the left). cons(j)
        should be itself a cell-array of size 1*4 taking this form for a
        factor of size m*n:
        {'constraint name', 'constraint parameter', m, n}

    'niter' - Number of iterations.
        Specifies the number of iterations to run.

    'init_facts' - Initialization of "facts".
        Specifies a starting point for the algorithm.

    Optional fields in PARAMS:
    --------------------------

    'init_lambda' - Initialization of "lambda".
        Specifies a starting point for the algorithm. The default value is 1

    'verbose' - Verbosity of the function. if verbose=1, the function
        outputs the error at each iteration. if verbose=0, the function runs
        in silent mode. The default value is 0.

    'update_way' - Way in which the factors are updated. If update_way = 1
        ,the factors are updated from right to left, and if update_way = 0,
        the factors are updated from left to right. The default value is 0.
    """
    # Setting optional parameters values
    init_lambda = params.get('init_lambda', 1)
    verbose = params.get('verbose', 0)
    update_way = params.get('update_way', 0)
    is_gpu = params.get('is_gpu', False)
    if is_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if params.n_facts != len(params.init_facts):
        raise Exception('Wrong initialization: params.nfacts and params.init_facts are in conflict')
    
    handles_cell = [lambda x: x] * params.n_facts
    for i in range(params.n_facts):
        cons = params.cons[i]
        if cons[0] == 'sp':
            handles_cell[i] = lambda x: prox_sp(x, cons[1])
        elif cons[0] == 'spcol':
            handles_cell[i] = lambda x: prox_spcol(x, cons[1])
        elif cons[0] == 'splin':
            handles_cell[i] = lambda x: prox_splin(x, cons[1])
        elif cons[0] == 'normcol':
            handles_cell[i] = lambda x: prox_normcol(x, cons[1])
        elif cons[0] == 'const':
            handles_cell[i] = lambda x: cons[1]
        elif cons[0] == 'sppos':
            handles_cell[i] = lambda x: prox_sp_pos(x, cons[1])
        else:
            raise Exception('The expressed type of constraint is not known')
    
    # Initialization
    lambda_ = init_lambda
    facts = params.init_facts
    for i, fact in enumerate(facts):
        if torch.is_tensor(fact):
            facts[i] = fact.to(device)
    
    params.data = params.data.to(device)
    X = params.data
    if update_way:
        maj = reversed(range(params.n_facts))
    else:
        maj = range(params.n_facts)
    
    for i in range(params.n_iter):
        for j in maj:
            if params.cons[j] == 'const':
                facts[j] = handles_cell[j](facts[j])
            else:
                L = facts[:j]
                R = facts[j+1:]
                if torch.isreal(X).all():
                    grad, LC = grad_comp(L, facts[j], R, params.data, lambda_, device)
                else:
                    grad, LC = grad_comp_cpx(L, facts[j], R, params, lambda_, device)

                c = LC * 1.001
                cons = params.cons[j]
                if cons[0] == 'l0pen':
                    pass
                elif cons[0] == 'l1pen':
                    pass
                else:
                    facts[j] = handles_cell[j](facts[j] - (1/c)*grad)
        
        lambda_ = torch.trace(mult_right(X.T, facts)) / torch.trace(mult_right(dvp(facts, device=device).T, facts))

        if verbose:
            rmse = torch.linalg.norm(X - lambda_*dvp(facts, device=device), 'fro') / math.sqrt(X.numel())
            print(f'Iter {i}, RMSE={rmse}')
    return lambda_, facts