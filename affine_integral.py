import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, logsumexp
from scipy.integrate import simpson
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def ho_wavefunction(n, omega=1.0):
    """
    Return the n-th harmonic oscillator wavefunction using stable evaluation.
    """
    def psi_n(x):
        # Use logarithmic normalization to avoid overflow
        # log(norm) = 0.25*log(omega/pi) - 0.5*log(2^n * n!)
        log_norm = 0.25 * np.log(omega / np.pi) - 0.5 * (n * np.log(2) + np.sum(np.log(np.arange(1, n+1))) if n > 0 else 0)

        # Evaluate Hermite polynomial using scipy's stable implementation
        xi = np.sqrt(omega) * x
        H_n = eval_hermite(n, xi)

        # Wavefunction
        return np.exp(log_norm) * H_n * np.exp(-xi**2 / 2)

    return psi_n

def compute_Z_matrix_numerical(M, a, b, omega=1.0):
    """
    Compute Z matrix numerically using direct integration with adaptive range.
    Z_{kn} = <phi_k | psi_n(ax+b)> = integral phi_k(x) * psi_n(ax+b) dx
    """
    # Adaptive integration range based on maximum quantum number
    # For HO, wavefunctions are significant within roughly sqrt(2n+1) in natural units
    max_n = M - 1
    x_range = max(10, 3 * np.sqrt(2 * max_n + 1) / np.sqrt(omega))

    # Adjust range for transformation parameters
    if abs(a) > 0:
        x_range = max(x_range, (x_range - abs(b)) / abs(a))

    # Use adaptive number of points
    n_points = max(10000, 200 * M)
    x_integration = np.linspace(-x_range, x_range, n_points)

    Z = np.zeros((M, M))

    for k in range(M):
        phi_k = ho_wavefunction(k, omega)(x_integration)
        for n in range(M):
            # Evaluate psi_n at transformed points
            x_transformed = a * x_integration + b
            psi_n = ho_wavefunction(n, omega)(x_transformed)

            # Compute overlap integral using Simpson's rule
            integrand = phi_k * psi_n
            Z[k, n] = simpson(integrand, x=x_integration)

    return Z

def compute_Z_matrix_gauss_hermite(M, a, b, omega=1.0):
    """
    Compute Z matrix using Gauss-Hermite quadrature with appropriate number of points.
    """
    # Use a reasonable number of quadrature points
    # For accurate integration of polynomials of degree up to 2M-1, we need M points
    # But for safety and transformed coordinates, use more
    n_points = min(max(100, 3 * M), 150)  # Cap at 150 to avoid overflow

    try:
        from numpy.polynomial.hermite import hermgauss
        x, w = hermgauss(n_points)
        # Transform from probabilist's Hermite (weight e^(-x^2/2)) to physicist's (weight e^(-x^2))
        x = x * np.sqrt(2)
        w = w * np.exp(x**2 / 2) / np.sqrt(2 * np.pi)

        # Scale for our harmonic oscillator frequency
        x = x / np.sqrt(omega)

    except Exception as e:
        # Fallback to direct integration if Gauss-Hermite fails
        print(f"Warning: Gauss-Hermite quadrature failed ({e}), using direct integration")
        return compute_Z_matrix_numerical(M, a, b, omega)

    Z = np.zeros((M, M))

    for k in range(M):
        phi_k = ho_wavefunction(k, omega)(x)
        for n in range(M):
            # Evaluate psi_n at transformed points
            x_transformed = a * x + b
            psi_n = ho_wavefunction(n, omega)(x_transformed)

            # The Hermite-Gauss quadrature includes the Gaussian weight
            # We need to include the remaining Gaussian factors from the wavefunctions
            # Since wavefunctions already include exp(-omega*x^2/2), we're good
            Z[k, n] = np.sum(w * phi_k * psi_n * np.exp(omega * x**2 / 2))

    return Z

def fit_transformed_wavefunction(n_target, a, b, omega, M, x_grid):
    """Fit psi_n(ax+b) using basis functions phi_k(x)"""
    # Compute Z matrix using direct integration (more stable)
    Z = compute_Z_matrix_numerical(M, a, b, omega)

    # Extract coefficients for target state
    coeffs = Z[:, n_target]

    # Reconstruct wavefunction
    psi_fit = np.zeros_like(x_grid, dtype=float)
    for k in range(M):
        psi_fit += coeffs[k] * ho_wavefunction(k, omega)(x_grid)

    # True wavefunction
    psi_true = ho_wavefunction(n_target, omega)(a * x_grid + b)

    return psi_true, psi_fit, coeffs


# Testing and plotting
print("="*70)
print("Testing wavefunction fitting using Z matrix")
print("="*70)

# Define grid - adaptive range
x_grid = np.linspace(-15, 15, 5000)

# Test parameters
omega = 1.0
a_vals = [0.7, 0.8, 0.9, 0.99, -1.00, 1.1]
b_vals = [0.0, 0.01, 0.2]
n_targets = [0, 1, 2, 3, 4, 5, 10, 15, 30, 60]
M_basis = 100

# Create figure for different quantum numbers
fig1, axes1 = plt.subplots(len(n_targets), 1, figsize=(10, 4*len(n_targets)))
if len(n_targets) == 1:
    axes1 = [axes1]

for idx, n in enumerate(n_targets):
    ax = axes1[idx]
    # Use first values from our test ranges
    a_test, b_test = a_vals[-1], b_vals[1]

    # Adaptive x range for this quantum number
    x_range = max(15, 3 * np.sqrt(2 * n + 1) / np.sqrt(omega))
    x_grid_n = np.linspace(-x_range, x_range, 5000)

    psi_true, psi_fit, coeffs = fit_transformed_wavefunction(
        n, a_test, b_test, omega, M_basis, x_grid_n
    )

    # Plot
    ax.plot(x_grid_n, psi_true, 'b-', linewidth=2, label=f'True ψ_{n}({a_test}x + {b_test})')
    ax.plot(x_grid_n, psi_fit, 'r--', linewidth=2, label=f'Fitted (M={M_basis})')
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('ψ(x)', fontsize=12)
    ax.set_title(f'n = {n}, a = {a_test}, b = {b_test}, ω = {omega}', fontsize=13)
    ax.legend(fontsize=10)

    # Compute error (L2 norm)
    error = np.sqrt(simpson((psi_true - psi_fit)**2, x=x_grid_n))
    max_error = np.max(np.abs(psi_true - psi_fit))
    ax.text(0.02, 0.98, f'L2 error: {error:.2e}\nMax error: {max_error:.2e}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    print(f"n={n:2d}: L2 error = {error:.3e}, Max error = {max_error:.3e}, "
          f"Non-zero coeffs = {np.sum(np.abs(coeffs) > 1e-10)}/{M_basis}")

plt.tight_layout()
plt.savefig('wavefunction_fit_by_n.png', dpi=150, bbox_inches='tight')
print("\nSaved: wavefunction_fit_by_n.png")
plt.close()

# Additional plot: convergence with basis size for high-n states
print("\n" + "="*70)
print("Testing convergence with basis size for high quantum numbers")
print("="*70)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()

test_cases = [
    (30, 1.0, 0.1),
    (60, 0.9, 0.0),
    (50, 1.1, 0.2),
    (40, 0.8, 0.05)
]

for idx, (n_test, a_test, b_test) in enumerate(test_cases):
    ax = axes2[idx]
    M_range = range(n_test + 10, min(n_test + 100, 150), 10)
    errors = []

    x_range = max(15, 3 * np.sqrt(2 * n_test + 1) / np.sqrt(omega))
    x_grid_test = np.linspace(-x_range, x_range, 3000)

    for M in M_range:
        psi_true, psi_fit, _ = fit_transformed_wavefunction(
            n_test, a_test, b_test, omega, M, x_grid_test
        )
        error = np.sqrt(simpson((psi_true - psi_fit)**2, x=x_grid_test))
        errors.append(error)

    ax.semilogy(list(M_range), errors, 'bo-', linewidth=2, markersize=6)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Basis size M', fontsize=11)
    ax.set_ylabel('L2 error', fontsize=11)
    ax.set_title(f'n={n_test}, a={a_test}, b={b_test}', fontsize=12)
    ax.axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='Target precision')
    ax.legend()

plt.tight_layout()
plt.savefig('convergence_high_n.png', dpi=150, bbox_inches='tight')
print("\nSaved: convergence_high_n.png")
plt.close()

print("\n" + "="*70)
print("All tests completed successfully!")
print("="*70)