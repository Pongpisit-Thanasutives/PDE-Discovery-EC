import math

def bic(loglik, k, n):
    if not (n > 0 and k >= 0):
        raise ValueError("Invalid input: n must be > 0 and k >= 0")
    return -2 * loglik + k * math.log(n)

def mbic(loglik, k, n, p, const=4):
    if not (n > 0 and k >= 0 and p > 0 and p/const > 1 and p/k >= 1):
        raise ValueError("Invalid input: ensure n > 0, k >= 0, p > 0, p/const > 1, and p/k >= 1")
    return bic(loglik, k, n) + 2 * k * math.log(p / const - 1)

def mbic2(loglik, k, n, p, const=4):
    if not (n > 0 and k >= 0 and p > 0 and p/const > 1 and p/k >= 1):
        raise ValueError("Invalid input: ensure n > 0, k >= 0, p > 0, p/const > 1, and p/k >= 1")
    
    if (k > p / 2) or (k > 3 * n / 4):
        return float('inf')
    
    if k < 150:
        penalty = 2 * math.log(math.factorial(k))
    else:
        penalty = 2 * sum(math.log(i) for i in range(1, k + 1))

    return mbic(loglik, k, n, p, const) - penalty

