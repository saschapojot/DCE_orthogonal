import mpmath as mp
import pandas as pd
from pathlib import Path
import sys
from multiprocessing import Pool
import itertools
from tqdm import tqdm

from datetime import datetime
#this script computes integral table using mpmath


mp.mp.dps=30
#python readCSV.py groupNum rowNum
#this script reads csv and creates directory
if len(sys.argv)!=3:
    print("wrong number of arguments")

groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])

#read parameters from csv
inParamFileName="./inParams/inParams"+str(groupNum)+".csv"
print("file name is "+inParamFileName)
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]
j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])
g0 = mp.mpf(oneRow.loc["g0"])
omega_m = mp.mpf(oneRow.loc["omegam"])  # Now reads full precision string into mpmath
omega_p = mp.mpf(oneRow.loc["omegap"])
omega_c = mp.mpf(oneRow.loc["omegac"])
er = mp.mpf(oneRow.loc["er"])
thetaCoef = mp.mpf(oneRow.loc["thetaCoef"])
theta = thetaCoef * mp.pi
#
N1 = int(oneRow.loc["N1"])
N2 = int(oneRow.loc["N2"])
tTot = mp.mpf(oneRow.loc["tTot"])
Q = int(oneRow.loc["Q"])

# derived quantities
# 1. Delta_m
Delta_m = omega_m - omega_p

#2. r
r=mp.log(er)
#3. e^{2r}
e2r=er**2
# #4. dt
dt=tTot/mp.mpf(Q)
tau=dt
#5.
lmd=(e2r-1.0/e2r)/(e2r+1.0/e2r)*Delta_m

#6. D
D=(lmd*mp.sin(theta))**2+omega_p**2
#7. mu
mu=lmd*mp.cos(theta)+Delta_m
#8. beta
beta=Delta_m-lmd*mp.cos(theta)
#9. Omega
Omega=mp.sqrt(beta*mu)
print("j1H="+str(j1H)+", j2H="+str(j2H)+", g0="+str(g0) \
      +", omega_m="+str(omega_m)+", omega_p="+str(omega_p) \
      +", omega_c="+str(omega_c)+", er="+str(er)+", thetaCoef="+str(thetaCoef)+f", N1={N1}, N2={N2}, tTot={tTot}, Q={Q}, tau={tau}")
print("\n" + "="*80)
print(f"{'DERIVED QUANTITY':<12}")
print("-" * 80)
##Derived Quantities
print(f"{'theta':<12} | {theta}")
print(f"{'Delta_m':<12} | {Delta_m}")
print(f"{'r':<12} | {r}")
print(f"{'lmd':<12} | {lmd}")
print(f"{'mu':<12} | {mu}")
print(f"{'beta':<12} | {beta}")
print(f"{'Omega':<12} | {Omega}")
print(f"{'D':<12} | {D}")
print("="*80 + "\n")
half=mp.mpf(0.5)
one_over_2=mp.mpf('1/2')
one_over_4=mp.mpf('1/4')
one_over_8=mp.mpf('1/8')
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


    rho_val=rho_func(x1,params)

    delta_val=delta_func(tau,params)

    val=rho_val*delta_val
    return val

def Z_tilde_summation_one_term(j,k,n1,n2,R,m1,m2,m3,m4,t,tau,params):
    """

    :param j:
    :param k:
    :param n1:
    :param n2:
    :param R:
    :param m1:
    :param m2:
    :param m3:
    :param m4:
    :param t:
    :return: one term in summation of Z tilde
    """
    omega_c = params['omega_c']
    Omega = params['Omega']
    alpha_val=alpha_func(tau, params)
    delta_val=delta_func(tau,params)

    pow_omega_c=(j - 2*m1 + n1 - 2*m2 + mp.mpf(1))/ mp.mpf(2)  + t
    part1=mp.power(omega_c,pow_omega_c)

    pow_Omega=(k + n2 - 2*R - 2*m3 - 2*m4) / mp.mpf(2)
    part2=mp.power(Omega,pow_Omega)

    pow_delta = k + n2 - 2*R - 2*m3 - 2*m4
    part3=mp.power(delta_val,pow_delta)

    pow_alpha_m1 = m3 + m4
    part4=mp.power(alpha_val**2 - 1,pow_alpha_m1)

    pow_alpha_p1 = -k - n2 - mp.mpf(0.5) + R + m3 + m4
    part5=mp.power(alpha_val**2 + 1,pow_alpha_p1)

    pow_sign=n2 + R + m1 + m2 + m3 + t
    part6=mp.power(-1,pow_sign)

    pow_alpha=k - 2*m3
    part7=mp.power(alpha_val,pow_alpha)

    # Factorial terms in denominator
    denom = (mp.factorial(R) * mp.factorial(m1) * mp.factorial(j - 2*m1) *
             mp.factorial(m2) * mp.factorial(n1 - 2*m2) *
             mp.factorial(m3) * mp.factorial(k - R - 2*m3) *
             mp.factorial(m4) * mp.factorial(n2 - R - 2*m4) *
             mp.factorial(t) * mp.factorial(k + n2 - 2*R - 2*m3 - 2*m4 - t))
    # Numerator factorials
    numer_fact = mp.sqrt(mp.factorial(j) * mp.factorial(n1) * mp.factorial(k) * mp.factorial(n2) / mp.pi) \
                 *mp.factorial(k+n2-2*R-2*m3-2*m4)

    part8=numer_fact/denom

    exp_sum_part1=one_over_4*(1+alpha_val**2)/(Omega*delta_val**2) \
                  *(one_over_2*Omega*delta_val**2/(1+alpha_val**2)-1)**2
    exp_sum_part2=-one_over_8*Omega*delta_val**2/(1+alpha_val**2)
    exp_sum=exp_sum_part1+exp_sum_part2
    part9=mp.exp(exp_sum)

    power_x = j - 2*m1 + n1 - 2*m2  + 2*t
    pow_val=(power_x+1)/mp.mpf(2)
    part10=(mp.sqrt((1+alpha_val**2)/Omega)*1/(mp.fabs(delta_val)*omega_c))**pow_val

    part11=mp.gamma((power_x+1) / mp.mpf(2))

    a_param = power_x / mp.mpf(2)
    z_param = -mp.sqrt((1+alpha_val**2)/Omega) \
              *(one_over_2*Omega*delta_val**2/(1+alpha_val**2)-1)*1/mp.fabs(delta_val)
    part12=mp.pcfu(a_param, z_param)

    power_2 = 2*R + mp.mpf(0.5)*j - 2*m1 + mp.mpf(0.5)*n1 - 2*m2 + t - mp.mpf(0.5)*k - mp.mpf(0.5)*n2 + mp.mpf(0.5)
    part13=mp.power(2, power_2)

    val=part1*part2*part3*part4\
        *part5*part6*part7*part8\
        *part9*part10*part11*part12\
        *part13

    return val

def one_Z_tilde_sequential(j,k,n1,n2,tau,params):
    # Check parity constraints
    if (j % 2) != (n1 % 2):
        return mp.mpf(0)
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
                            sum_total+=Z_tilde_summation_one_term(j,k,n1,n2,R,m1,m2,m3,m4,t,tau,params)


    return sum_total

def Z_tilde_summation_packed_params(packed_args):
    j,k,n1,n2,tau,params=packed_args
    Z_tilde_val=one_Z_tilde_sequential(j,k,n1,n2,tau,params)
    return (j,k,n1,n2,Z_tilde_val)




# Physical parameters
# omega_c = mp.mpf('1.5')
# omega_m = mp.mpf('1.1')
# omega_p = mp.mpf('0.8')
# Delta_m = omega_m - omega_p
# theta = mp.mpf('0.1')  # radians
# g0 = mp.mpf('0.2')  # Small coupling
# Derived parameters
# lmd=mp.mpf(0.9)*Delta_m
#
# mu = lmd * mp.cos(theta) + Delta_m
# beta = Delta_m - lmd * mp.cos(theta)
# Omega = mp.sqrt(beta * mu)
# D = lmd**2 * mp.sin(theta)**2 + omega_p**2
# params = {
#     'omega_c': omega_c,
#     'omega_m': omega_m,
#     'omega_p': omega_p,
#     'Delta_m': Delta_m,
#     'lmd': lmd,
#     'theta': theta,
#     'g0': g0,
#     'mu': mu,
#     'beta': beta,
#     'Omega': Omega,
#     'D': D
# }
#

# Time parameter
# tau = mp.mpf('0.1')  # Very small time

# j=2
# k=3
# n1=2
# n2=3
# one_list=[j,k,n1,n2,tau,params]
# param_list,val=Z_tilde_summation_packed_params(one_list)
# print(f"val={val}")
# Create an iterator (uses almost 0 memory)
t_table_start=datetime.now()
param_generator = (
    [j, k, n1, n2, tau, params]
    for j in range(0, N1)
    for k in range(0, N2)
    for n1 in range(0, N1)
    for n2 in range(0, N2)
)
# Calculate total iterations for tqdm
total_iterations = N1 * N2 * N1 * N2
results_list = []
paralel_num=24
with Pool(processes=paralel_num) as pool:
    # Use imap for memory efficiency
    # chunksize is crucial for speed if N1/N2 are large
    iterator = pool.imap(Z_tilde_summation_packed_params, param_generator, chunksize=1000)
    # Collect results
    for row in tqdm(iterator, total=total_iterations, desc="Computing Table"):
        results_list.append(row)

# 2. Create the Table
df = pd.DataFrame(results_list, columns=['j', 'k', 'n1', 'n2', 'Z_tilde'])
print(df.head())

t_table_end=datetime.now()

print(f"time: {t_table_end-t_table_start}")

outDir="./outData/group"+str(groupNum)+"/row"+str(rowNum)+"/"

Path(outDir).mkdir(exist_ok=True,parents=True)
out_csv_name=outDir+f"table_N1_{N1}_N2_{N2}.csv"
df.to_csv(out_csv_name, index=False)
print(f"File saved successfully to: {out_csv_name}")
