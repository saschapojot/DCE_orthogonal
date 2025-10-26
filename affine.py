import numpy as np
from scipy.special import loggamma, hyp2f1,eval_genlaguerre,factorial
import matplotlib.pyplot as plt
from scipy.special import hermite
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


#this script verifies the affine transformation's matrix elements


def compute_V(k, n, a):
    """
    Compute V_{kn}(a) where V_{kn}(a) = ∫ u_k(x) u_n(ax) dx

    Parameters:
    -----------
    k, n : int
        Indices for Hermite functions
    a : float
        Scaling parameter (should not be 1.0)

    Returns:
    --------
    V_{kn}(a) : float
        The matrix element
    """
    # Check parity - V_{kn} = 0 if n and k have different parity
    if (n % 2) != (k % 2):
        return 0.0

    # Compute lambda
    lambda_val = (a**2 - 1) / (a**2 + 1)

    # Compute p = (a^2 + 1)/2
    p = (a**2 + 1) / 2

    # Argument for hypergeometric function
    hyp_arg = -(1 - lambda_val**2) / lambda_val**2

    # Case 1: n and k both even
    if n % 2 == 0 and k % 2 == 0:
        # log|V_{kn}| for even n, k (Eq. 211)
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

        # Compute sign from Eq. (212)
        sign_lambda = np.sign(lambda_val)
        sign_hyp = np.sign(hyp_val)
        sign_V = sign_lambda**((n + k)//2) * (-1)**(k//2) * sign_hyp

        return np.exp(log_abs_V) * sign_V

    # Case 2: n and k both odd
    else:  # n % 2 == 1 and k % 2 == 1
        # log|V_{kn}| for odd n, k (Eq. 214)
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

        # Compute sign from Eq. (215)
        sign_lambda = np.sign(lambda_val)
        sign_hyp = np.sign(hyp_val)
        sign_V = sign_lambda**((n + k)//2 - 1) * (-1)**((k-1)//2) * sign_hyp

        return np.exp(log_abs_V) * sign_V


def compute_T(k, n, b):
    """
    Compute T_{kn}(b) where T_{kn}(b) = ∫ u_k(x) u_n(x - b) dx
    This is the Glauber displacement operator matrix element.

    Parameters:
    -----------
    k, n : int
        Indices for Hermite functions
    b : float
        Translation distance

    Returns:
    --------
    T_{kn}(b) : complex
        The matrix element
    """
    if b == 0:
        return 1.0 if n == k else 0.0

    b_squared_half = b**2 / 2.0

    # Case 1: k <= n
    if k <= n:
        # Eq. (318-319)
        term1 = 0.5 * (loggamma(k + 1) - loggamma(n + 1))
        term2 = (n - k) * np.log(np.abs(b) / np.sqrt(2))
        term3 = -b**2 / 4.0

        # Evaluate associated Laguerre polynomial L_k^(n-k)(b^2/2)
        laguerre_val = eval_genlaguerre(k, n - k, b_squared_half)
        term4 = np.log(np.abs(laguerre_val))

        log_abs_T = term1 + term2 + term3 + term4

        # Compute sign (Eq. 319)
        sign_b = np.sign(b) if b != 0 else 1
        sign_lag = np.sign(laguerre_val)
        sign_T = sign_b**(n - k) * sign_lag

        return np.exp(log_abs_T) * sign_T

    # Case 2: k > n
    else:
        # Eq. (321-322)
        term1 = 0.5 * (loggamma(n + 1) - loggamma(k + 1))
        term2 = (k - n) * np.log(np.abs(b) / np.sqrt(2))
        term3 = -b**2 / 4.0

        # Evaluate associated Laguerre polynomial L_n^(k-n)(b^2/2)
        laguerre_val = eval_genlaguerre(n, k - n, b_squared_half)
        term4 = np.log(np.abs(laguerre_val))

        log_abs_T = term1 + term2 + term3 + term4

        # Compute sign (Eq. 322)
        sign_b = np.sign(b) if b != 0 else 1
        sign_lag = np.sign(laguerre_val)
        sign_T = sign_b**(k - n) * (-1)**(k - n) * sign_lag

        return np.exp(log_abs_T) * sign_T



def construct_Z_matrix(M1, M2, a, b, omega, P_safety=100):
    """
   Construct the Z matrix for affine transformation of harmonic oscillator basis.

   Z_{kn} = ∫ ψ_k(x) ψ_n(ax + b) dx

   where ψ_j(x) = ω^(1/4) u_j(ω^(1/2) x) are harmonic oscillator eigenfunctions.

   According to Appendix D, Eq. (339):
   Z_{kn} = [V(a) T(-ω^(1/2) b)]_{kn}

   Parameters:
   -----------
   M1, M2 : int
       Matrix dimensions (output Z will be M1 × M2)
   a : float
       Scaling parameter (should not be 1.0)
   b : float
       Translation parameter
   omega : float
       Frequency parameter of harmonic oscillator
   P_safety : int, optional
       Safety margin for intermediate dimension (default: 50)

   Returns:
   --------
   Z : ndarray of shape (M1, M2), complex
       The affine transformation matrix
   """
    # Compute intermediate dimension
    P_tilde = max(M1, M2) + P_safety
    # Construct V matrix: M1 × P_tilde
    # V_{kp}(a) for k in [0, M1-1], p in [0, P_tilde-1]
    V = np.zeros((M1, P_tilde), dtype=np.float64)
    for k in range(M1):
        for p in range(P_tilde):
            V[k, p] = compute_V(k, p, a)

    # Construct T matrix: P_tilde × M2
    # T_{pn}(-ω^(1/2) b) for p in [0, P_tilde-1], n in [0, M2-1]

    b_scaled = -np.sqrt(omega) * b
    T = np.zeros((P_tilde, M2), dtype=np.complex128)
    for p in range(P_tilde):
        for n in range(M2):
            T[p, n] = compute_T(p, n, b_scaled)

    # Compute Z = V @ T (Eq. 341)
    Z = V @ T

    return Z



def construct_Z_matrix_even_odd(M1, M2, a, b, omega, P_safety=100):
    """
   Construct Z matrix using even/odd decomposition for efficiency.

   According to Appendix D (Eq. 356-357):
   Z^E = V^E @ T^E
   Z^O = V^O @ T^O

   Parameters:
   -----------
   M1, M2 : int
       Matrix dimensions
   a : float
       Scaling parameter (should not be 1.0)
   b : float
       Translation parameter
   omega : float
       Frequency parameter
   P_safety : int, optional
       Safety margin for intermediate dimension

   Returns:
   --------
   Z : ndarray of shape (M1, M2), complex
       The affine transformation matrix
   """
    # Compute intermediate dimension
    P_tilde = max(M1, M2) + P_safety
    # Compute even/odd dimensions
    M1_E = (M1 + 1) // 2  # ceil(M1/2)
    M1_O = M1 // 2         # floor(M1/2)
    P_tilde_E = (P_tilde + 1) // 2
    P_tilde_O = P_tilde // 2

    # Construct V^E: M1_E × P_tilde_E (even rows and columns of V)
    # V^E_{ij} = V_{2i, 2j}(a)
    V_E = np.zeros((M1_E, P_tilde_E), dtype=np.float64)
    for i in range(M1_E):
        for j in range(P_tilde_E):
            k = 2 * i
            p = 2 * j
            V_E[i, j] = compute_V(k, p, a)


    # Construct V^O: M1_O × P_tilde_O (odd rows and columns of V)
    # V^O_{ij} = V_{2i+1, 2j+1}(a)
    V_O = np.zeros((M1_O, P_tilde_O), dtype=np.float64)
    for i in range(M1_O):
        for j in range(P_tilde_O):
            k = 2 * i + 1
            p = 2 * j + 1
            V_O[i, j] = compute_V(k, p, a)


    # Construct T^E: P_tilde_E × M2 (even rows of T)
    # T^E_{pn} = T_{2p, n}(-ω^(1/2) b)
    b_scaled = -np.sqrt(omega) * b
    T_E = np.zeros((P_tilde_E, M2), dtype=np.complex128)
    for i in range(P_tilde_E):
        for n in range(M2):
            p = 2 * i
            T_E[i, n] = compute_T(p, n, b_scaled)

    # Construct T^O: P_tilde_O × M2 (odd rows of T)
    # T^O_{pn} = T_{2p+1, n}(-ω^(1/2) b)
    T_O = np.zeros((P_tilde_O, M2), dtype=np.complex128)
    for i in range(P_tilde_O):
        for n in range(M2):
            p = 2 * i + 1
            T_O[i, n] = compute_T(p, n, b_scaled)

    # Compute Z^E = V^E @ T^E (Eq. 356)
    Z_E = V_E @ T_E

    # Compute Z^O = V^O @ T^O (Eq. 357)
    Z_O = V_O @ T_O
    # Assemble full Z matrix (Eq. 361-362)
    Z = np.zeros((M1, M2), dtype=np.complex128)

    # Z_{2i, n} = Z^E_{i, n}
    for i in range(M1_E):
        Z[2*i, :] = Z_E[i, :]

    # Z_{2i+1, n} = Z^O_{i, n}
    for i in range(M1_O):
        Z[2*i+1, :] = Z_O[i, :]

    return Z
def ho_wavefunction(n, omega=1.0):
    """
    Return the n-th harmonic oscillator wavefunction.

    Parameters:
    -----------
    n : int
        Quantum number (n = 0, 1, 2, ...)
    omega : float
        Oscillator frequency

    Returns:
    --------
    function : callable
        A function that takes x and returns psi_n(x)
    """
    from scipy.special import hermite

    def psi_n(x):
        # Normalization constant
        norm = (omega / np.pi)**0.25 / np.sqrt(2**n * factorial(n))

        # Hermite polynomial
        H_n = hermite(n)

        # Wavefunction
        xi = np.sqrt(omega) * x
        return norm * H_n(xi) * np.exp(-xi**2 / 2)

    return psi_n
def compute_Z_matrix_numerical(M, a, b, omega=1.0, n_points=500):
    """
    Compute Z matrix numerically using Gauss-Hermite quadrature.
    This is more robust but slower.
    """
    from scipy.special import hermite

    # Use Gauss-Hermite quadrature
    x, w = np.polynomial.hermite.hermgauss(n_points)
    x = x * np.sqrt(2)  # Scale for our normalization
    w = w / np.sqrt(np.pi)  # Adjust weights

    Z = np.zeros((M, M))

    for k in range(M):
        phi_k = ho_wavefunction(k, omega)(x)
        for n in range(M):
            # Evaluate psi_n at transformed points
            x_transformed = a * x + b
            psi_n = ho_wavefunction(n, omega)(x_transformed)

            # Compute overlap integral
            Z[k, n] = np.sum(w * phi_k * psi_n)

    return Z
def harmonic_osc_wavefunction(n, x, omega):
    """
    Compute ψ_n(x) = ω^(1/4) u_n(ω^(1/2) x)
    where u_n is the normalized Hermite function.

    Parameters:
    -----------
    n : int
        Quantum number
    x : array_like
        Position values
    omega : float
        Frequency parameter

    Returns:
    --------
    psi : array_like
        Wavefunction values at x
    """
    y = np.sqrt(omega) * x
    # Normalization constant for Hermite functions
    norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    # Hermite polynomial
    Hn = hermite(n)
    # Complete wavefunction
    psi = omega**0.25 * norm * np.exp(-y**2 / 2.0) * Hn(y)
    return psi

def fit_transformed_wavefunction(n_target, a, b, omega, M, x_grid):
    """
    Fit ψ_n(ax + b) using basis functions ψ_k(x) via the Z matrix.

    The fitted function is: ψ_fit(x) = Σ_k Z_{kn} ψ_k(x)

    Parameters:
    -----------
    n_target : int
        Index of the target transformed wavefunction
    a : float
        Scaling parameter
    b : float
        Translation parameter
    omega : float
        Frequency parameter
    M_basis : int
        Number of basis functions to use
    x_grid : array_like
        Grid points for evaluation

    Returns:
    --------
    psi_true : array_like
        True values of ψ_n(ax + b)
    psi_fit : array_like
        Fitted values using Z matrix expansion
    coefficients : array_like
        Expansion coefficients Z_{:,n}
    """
    # Construct Z matrix using even/odd decomposition
    Z = compute_Z_matrix_numerical(M, a, b, omega)
    # Extract coefficients for n_target
    coefficients = Z[:, n_target]
    # Compute true transformed wavefunction
    psi_true = harmonic_osc_wavefunction(n_target, a * x_grid + b, omega)
    # Compute fitted wavefunction as linear combination
    psi_fit = np.zeros_like(x_grid, dtype=complex)
    for k in range(M_basis):
        psi_k = harmonic_osc_wavefunction(k, x_grid, omega)
        psi_fit += coefficients[k] * psi_k
    return psi_true, psi_fit, coefficients


# Testing and plotting
print("="*70)
print("Testing wavefunction fitting using Z matrix")
print("="*70)

# Define grid
x_grid = np.linspace(-5, 5, 500)

# Test parameters
omega = 1.0
a_vals = [0.7,0.8,0.9,0.99, 1.001, 1.1]
b_vals = [0.0, 0.01, 0.2]
n_targets = [0, 1,2, 3,4,5]
M_basis = 50

# Create figure for different quantum numbers
fig1, axes1 = plt.subplots(len(n_targets), 1, figsize=(10, 4*len(n_targets)))
if len(n_targets) == 1:
    axes1 = [axes1]

for idx, n in enumerate(n_targets):
    ax = axes1[idx]
    # Use first values from our test ranges
    a_test, b_test = a_vals[-2], b_vals[1]
    psi_true, psi_fit, coeffs = fit_transformed_wavefunction(
        n, a_test, b_test, omega, M_basis, x_grid
    )
    # Plot
    ax.plot(x_grid, psi_true, 'b-', linewidth=2, label=f'True ψ_{n}({a_test}x + {b_test})')
    ax.plot(x_grid, psi_fit.real, 'r--', linewidth=2, label=f'Fitted (M={M_basis})')
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('ψ(x)', fontsize=12)
    ax.set_title(f'n = {n}, a = {a_test}, b = {b_test}, ω = {omega}', fontsize=13)
    ax.legend(fontsize=10)

    # Compute error
    error = np.max(np.abs(psi_true - psi_fit.real))
    ax.text(0.02, 0.98, f'Max error: {error:.2e}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('wavefunction_fit_by_n.png', dpi=150, bbox_inches='tight')
print("\nSaved: wavefunction_fit_by_n.png")