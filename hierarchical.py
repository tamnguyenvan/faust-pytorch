import torch

from easydict import EasyDict
from palm4msa import palm4msa
from utils import dvp, nnzero_count


def hierarchical(params):
    """
      Hierarchical matrix factorization.
  [lambda, facts, errors] = hierarchical_fact(params) runs the hierarchical
  matrix factorization algorithm (Algorithm 3 of [1])on the specified
  input matrix, returning the factors in "facts" (cell array of sparse matrices), 
  the multiplicative scalar in "lambda" and the errors in "errors".


  Required fields in PARAMS:
  --------------------------

    'data' - Training data.
      A matrix to hierarchically factorize.

    'nfacts' - Number of factors.
      Specifies the desired number of factors.

    'cons' - Constraint sets.
      Specifies the constraint sets in which eafaust_params_palm.ch factor should lie. It
      should be a cell-array of size 2*(nfacts-1), where the jth columns
      sets the constraints for the jth factorization in two factors:
      cons(1,j) specifies the constraints for the left factor and
      cons(2,j) for the right factor. cons(i,j) should be itself a
      cell-array of size 1*4 taking this form for a factor of size m*n:
      {'constraint name', 'constraint parameter', m, n}


  Optional fields in PARAMS:
  --------------------------

    'niter1' - Number of iterations for the 2-factorisation.
      Specifies the desired number of iteration for the factorisations in
      2 factors. The default value is 500.

    'niter2' - Number of iterations for the global optimisation.
      Specifies the desired number of iteration for the global
      optimisation. The default value is 500.

    'fact_side' - Side to be factorized iteratively: 1-the left or
      0-the right. The default value is 0;

    'verbose' - Verbosity of the function. if verbose=1, the function
      outputs the error at each iteration. if verbose=0, the function runs
      in silent mode. The default value is 0.

    'update_way' - Way in which the factors are updated. If update_way = 1
      ,the factors are updated from right to left, and if update_way = 0,
      the factors are updated from left to right. The default value is 0.
    """
    # Setting parameters values
    n_iter1 = params.get('n_iter1', 500)
    n_iter2 = params.get('n_iter2', 500)
    verbose = params.get('verbose', 0)
    update_way = params.get('update_way', 0)
    fact_side = params.get('fact_side', 0)

    # Verify the validity of the constraints
    verif_size = params.data.size(0) == params.cons[0][0][2] and params.cons[0][0][3] \
        == params.cons[1][0][2] and params.data.size(1) == params.cons[1][0][3]
    
    for i in range(1, params.n_facts-1):
        if fact_side:
            verif_size = verif_size and params.cons[1][i-1][2] == params.cons[1][i][3] \
                and params.cons[0][i][3] == params.cons[1][i][2] and params.data.size(0) == params.cons[0][i][2]
        else:
            verif_size = verif_size and params.cons[0][i-1][3] == params.cons[0][i][2] \
                and params.cons[0][i][3] == params.cons[1][i][2] and params.data.size(1) == params.cons[1][i][3]
    
    if not verif_size:
        raise Exception('Size incompatibility in the constraints')
    
    if params.n_facts-1 != len(params.cons[0]):
        raise Exception('The number of constraints is in conflict with the number of factors')
    
    # Initialization
    lambda_ = 1
    facts = [[]] * params.n_facts
    Res = params.data
    errors = torch.zeros(params.n_facts-1, 2)

    for k in range(0, params.n_facts-1):
        cons = [params.cons[0][k], params.cons[1][k]]

        # Factorization in 2
        init_facts = [
            torch.zeros(cons[0][2], cons[0][3]), torch.eye(cons[1][2], cons[1][3])
        ]
        if update_way:
            init_facts = [
                torch.eye(cons[0][2], cons[0][3]), torch.zeros(cons[1][2], cons[1][3])
            ]
        params2 = EasyDict(
            n_iter=n_iter1,
            n_facts=2,
            data=Res,
            verbose=verbose,
            update_way=update_way,
            cons=[cons[0], cons[1]],
            init_facts=init_facts,
            init_lambda=1
        )
        lambda2, facts2 = palm4msa(params2)

        if fact_side:
            facts[2:] = facts[1:-1]
            facts[:2] = facts2
        else:
            facts[k:k+2] = facts2
        lambda_ = lambda_ * lambda2
        
        # Global optimization
        if fact_side:
            params3_cons = [cons[0]] + params.cons[1][k::-1]
        else:
            params3_cons = params.cons[0][:k+1] + [cons[1]]
        params3 = EasyDict(
            n_iter=n_iter2,
            n_facts=k+2,
            data=params.data,
            verbose=verbose,
            update_way=update_way,
            cons=params3_cons,
            init_facts=facts[:k+2],
            init_lambda=lambda_
        )
        lambda_, facts3 = palm4msa(params3)
        print('lambda', lambda_)
        facts[:k+2] = facts3
        if fact_side:
            Res = facts3[0]
        else:
            Res = facts3[k+1]
        
        errors[k, 0] = torch.linalg.norm(params.data - lambda_ * dvp(facts3)) / torch.linalg.norm(params.data)
        errors[k, 1] = nnzero_count(facts3) / params.data.numel()
    return lambda_, facts, errors