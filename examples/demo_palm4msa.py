import torch
from palm4msa import palm4msa
from easydict import EasyDict
from utils import dvp

X = torch.arange(100, dtype=torch.float64).view(10, 10)
m, n = X.size()
cons = [
    ['sp', 20, 10, 10],
    ['sp', 20, 10, 10],
]
init_facts = [
    torch.zeros(cons[0][2], cons[0][3], dtype=torch.float64),
    torch.eye(cons[1][2], cons[1][3], dtype=torch.float64)
]
params = EasyDict(
    n_iter=200,
    n_facts=2,
    data=X,
    verbose=1,
    update_way=0,
    cons=cons,
    init_facts=init_facts,
    init_lambda=1,
)

lambda_, facts = palm4msa(params)

print('X', X)
print('X_recon', dvp(facts))