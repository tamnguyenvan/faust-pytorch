def mult_right(x, D):
    """Multiplication by a factorized matrix.

    y = mult_right(x, D) computes the product y = x*D with D being in 
    factorized form. 
    """
    M = len(D)
    if M == 0:
        y = x
    else:
        y = x @ D[0]
        for i in range(1, M):
            y = y @ D[i]
    return y