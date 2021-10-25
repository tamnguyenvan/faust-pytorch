import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from hierarchical import hierarchical
from easydict import EasyDict
from utils import dvp


def main():
    X = torch.arange(100, dtype=torch.float32).view(10, 10)
    n_facts = 10
    cons = [
        [[]] * (n_facts-1),
        [[]] * (n_facts-1)
    ]
    cons[0][0] = ['sp', 20, 10, 10]
    cons[1][0] = ['sp', 20, 10, 10]
    for i in range(1, n_facts-1):
        cons[0][i] = ['sp', 20, 10, 10]
        cons[1][i] = ['sp', 20, 10, 10]
    is_gpu = False
    if torch.cuda.is_available() and args.gpu:
        is_gpu = True
    params = EasyDict(
        data=X,
        n_facts=n_facts,
        cons=cons,
        n_iter1=200,
        n_iter2=200,
        verbose=0,
        is_gpu=is_gpu,
    )

    lambda_, facts, errors = hierarchical(params)
    print('lambda', lambda_)
    print('facts', dvp(facts))
    print('errors', errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Use gpu if available')
    args = parser.parse_args()
    main()