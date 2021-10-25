import os
import sys
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from palm4msa import palm4msa
from easydict import EasyDict
from utils import dvp


def main():
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
    is_gpu = False
    if torch.cuda.is_available() and args.gpu:
        is_gpu = True
    params = EasyDict(
        n_iter=200,
        n_facts=2,
        data=X,
        verbose=args.verbose,
        update_way=0,
        cons=cons,
        init_facts=init_facts,
        init_lambda=1,
        is_gpu=is_gpu,
    )

    lambda_, facts = palm4msa(params)

    print('X', X)
    print('X_recon', dvp(facts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Use gpu if available')
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    args = parser.parse_args()
    main()