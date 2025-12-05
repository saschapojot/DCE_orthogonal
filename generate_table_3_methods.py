import mpmath as mp
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path
import sys
#this script generates integral table using 3 methods.
#python readCSV.py groupNum rowNum
#table is under the working dorectory

# Set high precision for mpmath
mp.dps = 40  # decimal places

if len(sys.argv)!=3:
    print("wrong number of arguments")

groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])

inParamFileName="./inParams/inParams"+str(groupNum)+".csv"

# print("file name is "+inParamFileName)
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]


j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])


g0=mp.mpf(oneRow.loc["g0"])
omega_m=mp.mpf(oneRow.loc["omegam"])
omega_p=mp.mpf(oneRow.loc["omegap"])
omega_c=mp.mpf(oneRow.loc["omegac"])
er=mp.mpf(oneRow.loc["er"])
thetaCoef=mp.mpf(oneRow.loc["thetaCoef"])
theta=thetaCoef*mp.pi
Delta_m = omega_m - omega_p
N1=int(oneRow.loc["N1"])
N2=int(oneRow.loc["N2"])
tTot=mp.mpf(oneRow.loc["tTot"])
Q=int(oneRow.loc["Q"])
r=mp.log(er)
lmd=mp.tanh(2*r)*Delta_m
mu = lmd * mp.cos(theta) + Delta_m
beta = Delta_m - lmd * mp.cos(theta)
Omega = mp.sqrt(beta * mu)
D = lmd**2 * mp.sin(theta)**2 + omega_p**2
print(f"j1H={j1H}, j2H={j2H}, g0={g0}, omega_m={omega_m}, omega_p={omega_p}, "
      f"omega_c={omega_c}, er={er}, r={r}, thetaCoef={thetaCoef}, theta={theta}, "
      f"Delta_m={Delta_m}, N1={N1}, N2={N2}, tTot={tTot}, Q={Q}, "
      f"lmd={lmd}, mu={mu}, beta={beta}, Omega={Omega}, D={D}")
outDir="./outData/group"+str(groupNum)+"/row"+str(rowNum)+"/"
Path(outDir).mkdir(exist_ok=True,parents=True)

params = {
    'omega_c': omega_c,
    'omega_m': omega_m,
    'omega_p': omega_p,
    'Delta_m': Delta_m,
    'lmd': lmd,
    'theta': theta,
    'g0': g0,
    'mu': mu,
    'beta': beta,
    'Omega': Omega,
    'D': D
}
def hermite_poly(n, x):
    """Compute Hermite polynomial H_n(x) using mpmath"""
    return mp.hermite(n, x)

def hermite_function(n, x):
    """Compute Hermite function u_n(x) = (2^n n! sqrt(pi))^(-1/2) exp(-x^2/2) H_n(x)"""
    norm = mp.power(2, n) * mp.factorial(n) * mp.sqrt(mp.pi)
    return mp.exp(-x**2 / 2) * hermite_poly(n, x) / mp.sqrt(norm)


def psi_c(j, x1, omega_c):
    """Cavity eigenfunction psi_j^c(x1) = omega_c^(1/4) u_j(omega_c^(1/2) x1)"""
    omega_c_sqrt = mp.sqrt(omega_c)
    omega_c_fourth = mp.power(omega_c, mp.mpf('0.25'))
    return omega_c_fourth * hermite_function(j, omega_c_sqrt * x1)


def psi_m(k, y2, Omega):
    """Phonon eigenfunction psi_k^m(y2) = Omega^(1/4) u_k(Omega^(1/2) y2)"""
    Omega_sqrt = mp.sqrt(Omega)
    Omega_fourth = mp.power(Omega, mp.mpf('0.25'))
    return Omega_fourth * hermite_function(k, Omega_sqrt * y2)


def alpha_func(tau,params):
    lmd = params['lmd']
    theta = params['theta']
    val=mp.exp(lmd*mp.sin(theta)*tau)
    return val

def rho_func(x1,params):
    omega_c = params['omega_c']
    val=omega_c*x1**2-mp.mpf("0.5")
    return val


def delta_func(tau,params):
    beta = params['beta']
    D = params['D']
    omega_p = params['omega_p']
    lmd = params['lmd']
    theta = params['theta']
    g0 = params['g0']
    alpha_val=alpha_func(tau,params)

    part0=-g0*mp.sqrt(2/beta)*lmd*mp.sin(theta)/D*alpha_val*mp.sin(omega_p*tau)

    part1=g0*mp.sqrt(2/beta)*omega_p/D*alpha_val*mp.cos(omega_p*tau)

    part2=-g0*mp.sqrt(2/beta)*omega_p/D

    return part0+part1+part2


def Delta_func(x1,tau,params):
    """Compute Delta(x1, tau) from equation (150)"""

    beta = params['beta']
    D = params['D']
    omega_p = params['omega_p']
    lmd = params['lmd']
    theta = params['theta']
    g0 = params['g0']

    alpha_val=alpha_func(tau,params)
    two=mp.mpf("2")
    rho_val=rho_func(x1,params)

    delta_val=delta_func(tau,params)

    val=rho_val*delta_val
    return val

def full_integrand(x1,y2,j,k,n1,n2,tau,params):
    """
    Compute the full x1,y2 integrand for RHS of equation (200):
    psi_j^c(x1) * psi_k^m(y2) * psi_n1^c(x1)  * psi_n2^m[alpha*y2 + Delta(x1,tau)]
    """
    Omega = params['Omega']
    lmd = params['lmd']
    theta = params['theta']
    alpha_val=alpha_func(tau,params)
    omega_p = params['omega_p']
    omega_c = params['omega_c']
    Delta_val=Delta_func(x1,tau,params)

    psi_c_j = psi_c(j, x1, omega_c)
    psi_m_k = psi_m(k, y2, Omega)
    psi_c_n1 = psi_c(n1, x1, omega_c)
    psi_m_n2=psi_m(n2,alpha_val*y2+Delta_val,Omega)

    return psi_c_j*psi_m_k*psi_c_n1*psi_m_n2


def compute_double_integral_numerical(j,k,n1,n2, tau, params, x1_max, y2_max, maxdegree=25):

    """
       Compute full double numerical integration

       Parameters:
       -----------

       maxdegree : int
           Maximum degree for Gaussian quadrature (default: 25)
       """

    print(f"    Integration bounds: x1 ∈ [{mp.nstr(-x1_max, 4)}, {mp.nstr(x1_max, 4)}], y2 ∈ [{mp.nstr(-y2_max, 4)}, {mp.nstr(y2_max, 4)}]")



    result = mp.quad(lambda x1:mp.quad(lambda y2: full_integrand(x1,y2,j,k,n1,n2, tau,params),[-y2_max,y2_max],maxdegree=maxdegree,points=[0]),
                     [-x1_max, x1_max],
                     maxdegree=maxdegree,points=[0])
    return result


def I_kn2_at_x1_func(k, n2, x1, tau, params, x1_max, y2_max, maxdegree=25):
    """
    Compute I_{kn2} at a specific x1 value using equation (165)
    This is the y2 integral evaluated analytically

    Parameters:
    -----------

    maxdegree : int
        Maximum degree for Gaussian quadrature when alpha > 1 (default: 25)
    """
    # omega_c = params['omega_c']
    Omega = params['Omega']
    # beta = params['beta']
    # D = params['D']
    # omega_p = params['omega_p']
    # lmd = params['lmd']
    # theta = params['theta']
    # g0 = params['g0']
    alpha_val=alpha_func(tau,params)
    # rho_val=rho_func(x1,params)
    Delta_val=Delta_func(x1,tau,params)
    # sigma = i (choosing positive imaginary unit)
    sigma = mp.mpc(0, 1)
    # Prefactor from equation (169)
    fact_part1=1 / mp.sqrt(mp.power(2, k + n2 - 1) * mp.factorial(k) * mp.factorial(n2))
    fact_part2=mp.exp(-mp.mpf('0.5') * Omega * Delta_val**2 / (1 + alpha_val **2))
    fact_part3=mp.power(alpha_val**2 - 1, (k + n2) / 2)/mp.power(alpha_val**2+1,(k+n2+1)/2)*sigma**n2
    prefactor=fact_part1*fact_part2*fact_part3
    # Sum over R
    sum_R = mp.mpc(0)
    for R in range(min(k, n2) + 1):
        binom_k_R = mp.binomial(k, R)
        binom_n2_R = mp.binomial(n2, R)
        factorial_R = mp.factorial(R)
        coeff = factorial_R * binom_k_R * binom_n2_R
        coeff *= mp.power(4*alpha_val/(sigma*mp.fabs(alpha_val**2-1)),R)

        arg1=-mp.sqrt(Omega)*alpha_val*Delta_val/mp.sqrt(mp.power(alpha_val, 4) - 1)
        arg2=-sigma*mp.sqrt(Omega)*Delta_val/mp.sqrt(mp.power(alpha_val, 4) - 1)
        H_k_minus_R =hermite_poly(k - R, arg1)
        H_n2_minus_R = hermite_poly(n2 - R, arg2)
        sum_R += coeff * H_k_minus_R * H_n2_minus_R
    result = prefactor * sum_R
    return result


def integral_using_feldheim(j,k,n1,n2,tau,params,x1_max, y2_max, maxdegree=25):
    """

    :param j:
    :param k:
    :param n1:
    :param n2:
    :param tau:
    :param params:
    :param x1_max:
    :param y2_max:
    :param maxdegree:
    :return: computing x1 integral after using feldheim
    """
    def integrand_x1(x1):
        # Analytical I_kn2
        I_kn2=I_kn2_at_x1_func(k, n2, x1, tau, params, x1_max, y2_max, maxdegree)
        # Cavity wavefunctions
        psi_c_j = psi_c(j, x1, params['omega_c'])
        psi_c_n1 = psi_c(n1, x1, params['omega_c'])
        return psi_c_j * psi_c_n1 * I_kn2

    result = mp.quad(integrand_x1, [-x1_max, x1_max], maxdegree=maxdegree,points=[0])
    return result


def Z_tilde_func(j,k,n1,n2,tau,params):
    """

    :param j:
    :param k:
    :param n1:
    :param n2:
    :param tau:
    :param params: analytical value of double integral
    :return:
    """
    # Check parity constraints
    if (j % 2) != (n1 % 2):
        return mp.mpf(0)


    omega_c = params['omega_c']
    Omega = params['Omega']
    lmd = params['lmd']
    theta = params['theta']
    beta = params['beta']
    D = params['D']
    omega_p = params['omega_p']
    g0 = params['g0']

    alpha_val=alpha_func(tau, params)
    delta_val=delta_func(tau,params)
    # Exponential term
    one_over_2=mp.mpf('1/2')
    one_over_4=mp.mpf('1/4')
    one_over_8=mp.mpf('1/8')
    exp_sum_part1=one_over_4*(1+alpha_val**2)/(Omega*delta_val**2) \
                  *(one_over_2*Omega*delta_val**2/(1+alpha_val**2)-1)**2
    exp_sum_part2=-one_over_8*Omega*delta_val**2/(1+alpha_val**2)
    exp_sum=exp_sum_part1+exp_sum_part2
    exp_part=mp.exp(exp_sum)
    # print(f"exp_part={exp_part}")


    # Common terms
    sum_total = mp.mpf(0)
    # Loop over R, m1, m2, m3, m4, t as in eq (200)
    min_k_n2 = min(k, n2)
    for R in range(0,min_k_n2 + 1):

        for m1 in range(0,j // 2 + 1):
            for m2 in range(0,n1 // 2 + 1):
                for m3 in range(0,(k - R) // 2 + 1):
                    for m4 in range(0,(n2 - R) // 2 + 1):
                        t_max = k + n2 - 2*R - 2*m3 - 2*m4
                        for t in range(0,t_max + 1):
                            # Compute all the terms from eq (200)
                            # Power of omega_c
                            power_omega_c = (j - 2*m1 + n1 - 2*m2 + mp.mpf(1))/ mp.mpf(2)  + t
                            # Power of Omega
                            power_Omega = (k + n2 - 2*R - 2*m3 - 2*m4) / mp.mpf(2)
                            # Power of delta (which is now δ(τ))
                            power_delta = k + n2 - 2*R - 2*m3 - 2*m4
                            # Power of (alpha^2 - 1)
                            power_alpha_m1 = m3 + m4
                            # Power of (alpha^2 + 1)
                            power_alpha_p1 = -k - n2 - mp.mpf(0.5) + R + m3 + m4
                            # Power of 2
                            power_2 = 2*R + mp.mpf(0.5)*j - 2*m1 + mp.mpf(0.5)*n1 - 2*m2 + t - mp.mpf(0.5)*k - mp.mpf(0.5)*n2 + mp.mpf(0.5)

                            # Power of alpha (for alpha^(k-2m3))
                            power_alpha = k - 2*m3
                            # Sign from (-1)^(n2+R+m1+m2+m3+t)
                            sign = mp.power(-1, n2 + R + m1 + m2 + m3 + t)
                            # Factorial terms in denominator
                            denom = (mp.factorial(R) * mp.factorial(m1) * mp.factorial(j - 2*m1) *
                                     mp.factorial(m2) * mp.factorial(n1 - 2*m2) *
                                     mp.factorial(m3) * mp.factorial(k - R - 2*m3) *
                                     mp.factorial(m4) * mp.factorial(n2 - R - 2*m4) *
                                     mp.factorial(t) * mp.factorial(k + n2 - 2*R - 2*m3 - 2*m4 - t))
                            # Numerator factorials
                            numer_fact = mp.sqrt(mp.factorial(j) * mp.factorial(n1) * mp.factorial(k) * mp.factorial(n2) / mp.pi) \
                                         *mp.factorial(k+n2-2*R-2*m3-2*m4)

                            # Compute the main coefficient
                            coeff = (numer_fact / denom *
                                     mp.power(omega_c, power_omega_c) *
                                     mp.power(Omega, power_Omega) *
                                     mp.power(delta_val, power_delta) *
                                     mp.power(alpha_val**2 - 1, power_alpha_m1) *
                                     mp.power(alpha_val**2 + 1, power_alpha_p1) *
                                     mp.power(2, power_2) *
                                     mp.power(alpha_val, power_alpha) *
                                     sign)
                            power_x = j - 2*m1 + n1 - 2*m2  + 2*t
                            pow_val=(power_x+1)/mp.mpf(2)
                            pow_term=(mp.sqrt((1+alpha_val**2)/Omega)*1/(mp.fabs(delta_val)*omega_c))**pow_val

                            a_param = power_x / mp.mpf(2)
                            z_param = -mp.sqrt((1+alpha_val**2)/Omega) \
                                      *(one_over_2*Omega*delta_val**2/(1+alpha_val**2)-1)*1/np.abs(delta_val)
                            U_term=mp.pcfu(a_param, z_param)
                            gm_val=mp.gamma((power_x+1) / mp.mpf(2))
                            prod_val=coeff*pow_term*gm_val*U_term
                            sum_total+=prod_val
    # print(f"sum_total={sum_total}")
    sum_total*=exp_part

    return sum_total
def estimate_integration_bounds(j, k, n1, n2, params, tau, safety_factor=3.0):
    """
    Estimate appropriate integration bounds based on quantum numbers and parameters

    For harmonic oscillator, classical turning points are approximately at ±sqrt((2n+1)/omega)
    We add a safety factor to ensure we capture the tails
    """
    omega_c = params['omega_c']
    Omega = params['Omega']

    # Maximum quantum number for cavity
    n_max_cavity = max(j, n1)
    # Estimate x1 range based on cavity turning points
    x1_classical = mp.sqrt((2 * n_max_cavity + 1) / omega_c)
    x1_max = float(x1_classical * safety_factor)

    # Maximum quantum number for phonon
    n_max_phonon = max(k, n2)
    # Estimate y2 range based on phonon turning points
    y2_classical = mp.sqrt((2 * n_max_phonon + 1) / Omega)

    # Account for the transformation alpha*y2 + Delta
    alpha_val = alpha_func(tau, params)
    # Rough estimate of maximum Delta (depends on x1_max)
    delta_val_max = float(mp.fabs(delta_func(tau, params)))
    rho_max = omega_c * x1_max**2 + 0.5  # upper bound on |rho|
    Delta_max = rho_max * delta_val_max

    # Account for scaling and translation in y2
    y2_max_base = float(y2_classical * safety_factor)
    # Add buffer for the transformation
    y2_max = y2_max_base * float(alpha_val) + Delta_max + y2_max_base
    # --- ADD THESE LINES HERE ---
    x1_max += 2.0
    y2_max += 2.0
    return x1_max, y2_max

def adaptive_compute_integral(j, k, n1, n2, tau, params, method='method3',
                              initial_x1_max=None, initial_y2_max=None,
                              maxdegree=25, tolerance=1e-10, max_iterations=5):
    """
    Compute integral with adaptive integration bounds

    Parameters:
    -----------
    method : str
        'method1', 'method2', or 'method3'
    tolerance : float
        Relative tolerance for convergence
    max_iterations : int
        Maximum number of bound enlargements

    Returns:
    --------
    result, x1_max, y2_max, elapsed_time
    """

    start_time = datetime.now()

    # Get initial bounds estimate
    if initial_x1_max is None or initial_y2_max is None:
        x1_max, y2_max = estimate_integration_bounds(j, k, n1, n2, params, tau)
    else:
        x1_max, y2_max = initial_x1_max, initial_y2_max

    # For method3 (analytical), no integration bounds needed
    if method == 'method3':
        result = Z_tilde_func(j, k, n1, n2, tau, params)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        return result, x1_max, y2_max, elapsed_time

    # Initialize
    prev_result = None
    enlargement_factor = 1.5

    for iteration in range(max_iterations):
        # Compute integral with current bounds
        if method == 'method1':
            result = compute_double_integral_numerical(
                j, k, n1, n2, tau, params, x1_max, y2_max, maxdegree
            )
        elif method == 'method2':
            result = integral_using_feldheim(
                j, k, n1, n2, tau, params, x1_max, y2_max, maxdegree
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Check convergence
        if prev_result is not None:
            # Compute relative difference
            result_val = float(result.real) if hasattr(result, 'real') else float(result)
            prev_val = float(prev_result.real) if hasattr(prev_result, 'real') else float(prev_result)

            if abs(result_val) > 1e-15:  # Avoid division by zero
                rel_diff = abs(result_val - prev_val) / abs(result_val)
            else:
                rel_diff = abs(result_val - prev_val)

            if rel_diff < tolerance:
                # Converged
                elapsed_time = (datetime.now() - start_time).total_seconds()
                return result, x1_max, y2_max, elapsed_time

        # Enlarge bounds for next iteration
        prev_result = result
        x1_max *= enlargement_factor
        y2_max *= enlargement_factor

    # Did not converge within max_iterations
    print(f"    Warning: [j={j}, k={k}, n1={n1}, n2={n2}] did not converge within {max_iterations} iterations")
    elapsed_time = (datetime.now() - start_time).total_seconds()
    return result, x1_max, y2_max, elapsed_time

def compute_all_integrals_adaptive(tau, params, N1, N2, maxdegree=25,
                                   use_adaptive=True, tolerance=1e-10):
    """
    Compute integrals for all combinations with adaptive integration bounds

    Parameters:
    -----------
    use_adaptive : bool
        If True, use adaptive bounds. If False, use estimated bounds directly.
    tolerance : float
        Convergence tolerance for adaptive method
    """
    results = []

    total_combinations = N1 * N2 * N1 * N2
    current = 0

    print(f"\nComputing {total_combinations} integrals...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"tau = {mp.nstr(tau, 6)}")
    print(f"Adaptive bounds: {use_adaptive}, maxdegree={maxdegree}")
    if use_adaptive:
        print(f"Convergence tolerance: {tolerance}")
    print("="*80)

    overall_start_time = datetime.now()

    # Track bounds statistics
    x1_bounds = []
    y2_bounds = []

    # Track timing statistics
    method1_times = []
    method2_times = []
    method3_times = []

    for j in range(N1):
        for k in range(N2):
            for n1 in range(N1):
                for n2 in range(N2):
                    current += 1
                    integral_start_time = datetime.now()

                    # Progress indicator
                    if current % 10 == 0 or current == 1:
                        elapsed = (datetime.now() - overall_start_time).total_seconds()
                        rate = current / elapsed if elapsed > 0 else 0
                        eta = (total_combinations - current) / rate if rate > 0 else 0
                        eta_str = f"{int(eta//3600):02d}:{int((eta%3600)//60):02d}:{int(eta%60):02d}"
                        print(f"Progress: {current}/{total_combinations} "
                              f"({100*current/total_combinations:.1f}%), "
                              f"Rate: {rate:.2f} integrals/s, "
                              f"ETA: {eta_str}")

                    # Check parity constraint
                    if (j % 2) != (n1 % 2):
                        method1 = mp.mpf(0)
                        method2 = mp.mpc(0)
                        method3 = mp.mpf(0)
                        x1_used = 0
                        y2_used = 0
                        time_method1 = 0
                        time_method2 = 0
                        time_method3 = 0
                    else:
                        # Get initial bounds estimate
                        x1_init, y2_init = estimate_integration_bounds(j, k, n1, n2, params, tau)

                        if use_adaptive:
                            # Adaptive integration
                            method1, x1_m1, y2_m1, time_method1 = adaptive_compute_integral(
                                j, k, n1, n2, tau, params, 'method1',
                                x1_init, y2_init, maxdegree, tolerance
                            )
                            method2, x1_m2, y2_m2, time_method2 = adaptive_compute_integral(
                                j, k, n1, n2, tau, params, 'method2',
                                x1_init, y2_init, maxdegree, tolerance
                            )
                            x1_used = max(x1_m1, x1_m2)
                            y2_used = max(y2_m1, y2_m2)
                        else:
                            # Use estimated bounds directly
                            x1_used = x1_init
                            y2_used = y2_init

                            t1_start = datetime.now()
                            method1 = compute_double_integral_numerical(
                                j, k, n1, n2, tau, params, x1_used, y2_used, maxdegree
                            )
                            time_method1 = (datetime.now() - t1_start).total_seconds()

                            t2_start = datetime.now()
                            method2 = integral_using_feldheim(
                                j, k, n1, n2, tau, params, x1_used, y2_used, maxdegree
                            )
                            time_method2 = (datetime.now() - t2_start).total_seconds()

                        # Method 3 is always analytical (no bounds needed)
                        method3, _, _, time_method3 = adaptive_compute_integral(
                            j, k, n1, n2, tau, params, 'method3'
                        )

                        x1_bounds.append(x1_used)
                        y2_bounds.append(y2_used)
                        method1_times.append(time_method1)
                        method2_times.append(time_method2)
                        method3_times.append(time_method3)

                    # Total time for this integral
                    integral_elapsed = (datetime.now() - integral_start_time).total_seconds()

                    # Convert to float for comparison
                    val1 = float(method1.real) if hasattr(method1, 'real') else float(method1)
                    val2 = float(method2.real)
                    val3 = float(method3.real) if hasattr(method3, 'real') else float(method3)

                    # Compute differences
                    diff_12 = abs(val1 - val2)
                    diff_23 = abs(val2 - val3)
                    diff_13 = abs(val1 - val3)

                    results.append({
                        'j': j,
                        'k': k,
                        'n1': n1,
                        'n2': n2,
                        'method1_real': val1,
                        'method2_real': val2,
                        'method3_real': val3,
                        'method1_imag': float(method1.imag) if hasattr(method1, 'imag') else 0.0,
                        'method2_imag': float(method2.imag),
                        'method3_imag': float(method3.imag) if hasattr(method3, 'imag') else 0.0,
                        'abs_diff_12': diff_12,
                        'abs_diff_23': diff_23,
                        'abs_diff_13': diff_13,
                        'max_diff': max(diff_12, diff_23, diff_13),
                        'x1_max_used': x1_used,
                        'y2_max_used': y2_used,
                        'time_method1_sec': time_method1,
                        'time_method2_sec': time_method2,
                        'time_method3_sec': time_method3,
                        'time_total_sec': integral_elapsed
                    })

    overall_elapsed_time = (datetime.now() - overall_start_time).total_seconds()
    end_time = datetime.now()

    print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed time: {int(overall_elapsed_time//3600):02d}:{int((overall_elapsed_time%3600)//60):02d}:{int(overall_elapsed_time%60):02d} (HH:MM:SS)")
    print(f"Total elapsed time: {overall_elapsed_time/60:.2f} minutes")
    print(f"Average time per integral: {overall_elapsed_time/total_combinations:.3f} seconds")

    if x1_bounds:
        print(f"\nIntegration bounds statistics:")
    print(f"  x1_max: min={float(min(x1_bounds)):.2f}, max={float(max(x1_bounds)):.2f}, mean={np.mean([float(x) for x in x1_bounds]):.2f}")
    print(f"  y2_max: min={float(min(y2_bounds)):.2f}, max={float(max(y2_bounds)):.2f}, mean={np.mean([float(x) for x in y2_bounds]):.2f}")

    if method1_times:
        print(f"\nTiming statistics per method (non-zero integrals only):")
        print(f"  Method 1 (double numerical): mean={np.mean(method1_times):.3f}s, max={max(method1_times):.3f}s")
        print(f"  Method 2 (Feldheim + num):   mean={np.mean(method2_times):.3f}s, max={max(method2_times):.3f}s")
        print(f"  Method 3 (fully analytical): mean={np.mean(method3_times):.3f}s, max={max(method3_times):.3f}s")

    return pd.DataFrame(results)


# Time parameter
tau = mp.mpf('0.1')  # Very small time

# Record overall start time
computation_start = datetime.now()
print(f"\nComputation started at: {computation_start.strftime('%Y-%m-%d %H:%M:%S')}")

# Compute all integrals with adaptive bounds
df_results = compute_all_integrals_adaptive(
    tau, params, N1, N2,
    maxdegree=25,
    use_adaptive=True,  # Set to False to use estimated bounds without adaptation
    tolerance=1e-10
)

# Save results with timestamp
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = outDir + f"integral_table_tau_{mp.nstr(tau, 6)}_{timestamp_str}.csv"
df_results.to_csv(output_file, index=False, float_format='%.15e')
print(f"\nResults saved to: {output_file}")

# Also save a timing summary
timing_summary_file = outDir + f"timing_summary_{timestamp_str}.txt"
with open(timing_summary_file, 'w') as f:
    f.write(f"Computation Start: {computation_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Computation End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Time: {(datetime.now() - computation_start).total_seconds()/60:.2f} minutes\n")
    f.write(f"\nParameters:\n")
    f.write(f"N1={N1}, N2={N2}, Q={Q}, tau={mp.nstr(tau, 6)}\n")
    f.write(f"Total integrals: {len(df_results)}\n")
    f.write(f"\nTiming per integral:\n")
    f.write(f"Mean total: {df_results['time_total_sec'].mean():.3f}s\n")
    f.write(f"Mean method1: {df_results['time_method1_sec'].mean():.3f}s\n")
    f.write(f"Mean method2: {df_results['time_method2_sec'].mean():.3f}s\n")
    f.write(f"Mean method3: {df_results['time_method3_sec'].mean():.3f}s\n")

print(f"Timing summary saved to: {timing_summary_file}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS:")
print("="*80)
print(f"Total integrals computed: {len(df_results)}")
print(f"Non-zero integrals: {(df_results['method1_real'].abs() > 1e-10).sum()}")
print(f"\nMaximum differences between methods:")
print(f"  |method1 - method2|: {df_results['abs_diff_12'].max():.2e}")
print(f"  |method2 - method3|: {df_results['abs_diff_23'].max():.2e}")
print(f"  |method1 - method3|: {df_results['abs_diff_13'].max():.2e}")
print(f"\nMean differences between methods:")
print(f"  |method1 - method2|: {df_results['abs_diff_12'].mean():.2e}")
print(f"  |method2 - method3|: {df_results['abs_diff_23'].mean():.2e}")
print(f"  |method1 - method3|: {df_results['abs_diff_13'].mean():.2e}")

# Show worst agreements
print(f"\nTop 5 worst agreements:")
worst_5 = df_results.nlargest(5, 'max_diff')[['j', 'k', 'n1', 'n2', 'method1_real', 'method2_real', 'method3_real', 'max_diff', 'x1_max_used', 'y2_max_used', 'time_total_sec']]
print(worst_5.to_string(index=False))

# Show slowest integrals
print(f"\nTop 5 slowest integrals:")
slowest_5 = df_results.nlargest(5, 'time_total_sec')[['j', 'k', 'n1', 'n2', 'time_method1_sec', 'time_method2_sec', 'time_method3_sec', 'time_total_sec']]
print(slowest_5.to_string(index=False))

print("\n" + "="*80)