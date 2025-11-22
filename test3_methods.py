import mpmath as mp
import numpy as np
from datetime import datetime

# Set high precision for mpmath
mp.dps = 30  # decimal places

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

    part1=-g0*mp.sqrt(two/beta)*lmd*mp.sin(theta)/D*rho_val*alpha_val\
         *mp.sin(omega_p*tau)
    part2=g0*mp.sqrt(two/beta)*omega_p/D*rho_val*alpha_val\
          *mp.cos(omega_p*tau)

    part3=-g0*mp.sqrt(two/beta)*omega_p/D*rho_val

    val=part1+part2+part3
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



    result = mp.quad(lambda x1:mp.quad(lambda y2: full_integrand(x1,y2,j,k,n1,n2, tau,params),[-y2_max,y2_max],maxdegree=maxdegree),
                     [-x1_max, x1_max],
                     maxdegree=maxdegree)
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

    result = mp.quad(integrand_x1, [-x1_max, x1_max], maxdegree=maxdegree)
    return result

def delta_func(tau,params):
    omega_c = params['omega_c']
    Omega = params['Omega']
    lmd = params['lmd']
    theta = params['theta']
    beta = params['beta']
    D = params['D']
    omega_p = params['omega_p']
    g0 = params['g0']
    alpha_val=alpha_func(tau, params)
    part1=-g0*mp.sqrt(2/beta)*lmd*mp.sin(theta)/D*alpha_val*mp.sin(omega_p*tau)
    part2=g0*mp.sqrt(2/beta)*omega_p/D*alpha_val*mp.cos(omega_p*tau)
    part3=-g0*mp.sqrt(2/beta)*omega_p/D

    val=part1+part2+part3
    return val



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
    if (k % 2) != (n2 % 2):
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
    print(f"exp_part={exp_part}")


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
                            numer_fact = mp.sqrt(mp.factorial(j) * mp.factorial(n1) * mp.factorial(k) * mp.factorial(n2) / mp.pi)\
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
                            power_x = j - 2*m1 + n1 - 2*m2 + k - 2*R - 2*m3 + n2 - 2*m4 + 2*t
                            pow_val=(power_x+1)/mp.mpf(2)
                            pow_term=(mp.sqrt((1+alpha_val**2)/Omega)*1/(mp.fabs(delta_val)*omega_c))**pow_val

                            a_param = power_x / mp.mpf(2)
                            z_param = -mp.sqrt((1+alpha_val**2**2)/Omega) \
                                      *(one_over_2*Omega*delta_val**2/(1+alpha_val**2)-1)*1/np.abs(delta_val)
                            U_term=mp.pcfu(a_param, z_param)
                            gm_val=mp.gamma((power_x+1) / mp.mpf(2))
                            prod_val=coeff*pow_term*gm_val*U_term
                            sum_total+=prod_val
    print(f"sum_total={sum_total}")
    sum_total*=exp_part

    return sum_total







# Physical parameters
omega_c = mp.mpf('1.5')
omega_m = mp.mpf('1.1')
omega_p = mp.mpf('0.8')
Delta_m = omega_m - omega_p
theta = mp.mpf('0.1')  # radians
g0 = mp.mpf('0.2')  # Small coupling
# Derived parameters
lmd=0.9*Delta_m

mu = lmd * mp.cos(theta) + Delta_m
beta = Delta_m - lmd * mp.cos(theta)
Omega = mp.sqrt(beta * mu)
D = lmd**2 * mp.sin(theta)**2 + omega_p**2
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


# Time parameter
tau = mp.mpf('2')  # Very small time

j=1
k=1
n1=1
n2=1
x1_max=7
y2_max=15
max_deg=25

rst1=compute_double_integral_numerical(j,k,n1,n2,tau,params,x1_max,y2_max,max_deg)
rst2=integral_using_feldheim(j,k,n1,n2,tau,params,x1_max,y2_max,max_deg)
rst3=Z_tilde_func(j,k,n1,n2,tau,params)
print(f"rst1={rst1}, rst2={rst2}, rst3={rst3}")