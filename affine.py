import numpy as np
from scipy.special import loggamma, hyp2f1
from scipy.special import factorial, gamma


#this script verifies the affine transformation's matrix elements
# def log_pochhammer(x, n):
#     """Compute log of Pochhammer symbol (x)_n"""
#     if n == 0:
#         return 0.0
#     return loggamma(x + n) - loggamma(x)
# def log_factorial(n):
#     """Compute log(n!)"""
#     return loggamma(n + 1)

def compute_V(n, k, a):
    """
    Compute log|V_{kn}(a)| where V_{kn}(a) = ∫ u_k(x) u_n(ax) dx
    and returns V_{kn}(a)
    Parameters:
    -----------
    n, k : int
        Indices for Hermite functions
    a : float
        Scaling parameter

    Returns: V_{kn}(a)
    """
    # Check parity - V_{kn} = 0 if n and k have different parity
    if (n % 2) != (k % 2):
        return 0

    # Compute lambda
    lambda_val = (a**2 - 1) / (a**2 + 1)

    # Compute p = (a^2 + 1)/2
    p = (a**2 + 1) / 2
    # Argument for hypergeometric function
    hyp_arg = -(1 - lambda_val**2) / lambda_val**2

    # Case 1: n and k both even
    if n % 2 == 0 and k % 2 == 0:
        # log|V_{kn}| for even n, k
        term1 = -(n + k) / 2 * np.log(2)
        term2 = -0.5 * loggamma(n + 1)
        term3 = -0.5 * loggamma(k + 1)
        term4 = -0.5 * np.log(p)
        term5 = (n + k) * np.log(2 * np.abs(lambda_val))
        term6 = loggamma(n + 0.5) - loggamma(0.5)
        term7 = loggamma(k + 0.5) - loggamma(0.5)

        # Compute hypergeometric function
        hyp_val = hyp2f1(-n/2, -k/2, 0.5, hyp_arg)
        term8 = np.log(np.abs(hyp_val))

        log_abs_V = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

        # Compute sign
        sign_lambda = np.sign(lambda_val)
        sign_hyp = np.sign(hyp_val)
        sign_V = sign_lambda**((n + k)//2) * (-1)**(k//2) * sign_hyp

        return np.exp(log_abs_V)*sign_V

    # Case 2: n and k both odd
    else:  # n % 2 == 1 and k % 2 == 1
        # log|V_{kn}| for odd n, k
        term1 = -(n + k) / 2 * np.log(2)
        term2 = -0.5 * loggamma(n + 1)
        term3 = -0.5 * loggamma(k + 1)
        term4 = -0.5 * np.log(p)
        term5 = (n + k - 1) * np.log(2)
        term6 = 0.5 * np.log(1 - lambda_val**2)
        term7 = ((n + k) / 2 - 1) * np.log(np.abs(lambda_val))
        term8 = loggamma((n + 2) / 2) - loggamma(1.5)
        term9 = loggamma((k + 2) / 2) - loggamma(1.5)
        # Compute hypergeometric function
        hyp_val = hyp2f1(-(n-1)/2, -(k-1)/2, 1.5, hyp_arg)
        term10 = np.log(np.abs(hyp_val))
        log_abs_V = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
        # Compute sign
        sign_lambda = np.sign(lambda_val)
        sign_hyp = np.sign(hyp_val)
        sign_V = sign_lambda**((n + k)//2 - 1) * (-1)**((k-1)//2) * sign_hyp

        return np.exp(log_abs_V) * sign_V




