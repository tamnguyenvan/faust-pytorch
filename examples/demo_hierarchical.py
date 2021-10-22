import torch
from hierarchical import hierarchical
from easydict import EasyDict
from utils import dvp


X = torch.arange(100, dtype=torch.float32).view(10, 10)
n_facts = 3
cons = [
    [[]] * (n_facts-1),
    [[]] * (n_facts-1)
]
cons[0][0] = ['sp', 20, 10, 10]
cons[1][0] = ['sp', 20, 10, 10]
for i in range(1, n_facts-1):
    cons[0][i] = ['sp', 20, 10, 10]
    cons[1][i] = ['sp', 20, 10, 10]
params = EasyDict(
    data=X,
    n_facts=n_facts,
    cons=cons,
    n_iter1=200,
    n_iter2=200,
    verbose=0
)

lambda_, facts, errors = hierarchical(params)
print('lambda', lambda_)
print('facts', facts)
print('errors', errors)