

def mult_left(D, x):
    """Multiplication by a factorized matrix.

    y = mult_left(D,x) computes the product y = D*x with D being in 
    factorized form. 
    """
    M = len(D)
    if M == 0:
        y = x
    else:
        y = D[M-1] @ x
        for i in range(M-2, -1, -1):
            y = D[i] @ y
    return y