from sympy import *
from sympy.simplify.fu import TR5,TR11
#this script verifies if analytical form of Z is correct
######################## symbolic
omega_c,omega_m,omega_p=symbols("omega_c,omega_m,omega_p",cls=Symbol,real=True,positive =True)
Delta_m = omega_m - omega_p
theta = symbols('theta',cls=Symbol,real=True)  # radians
g0,lmd,tau =symbols("g0,lambda,tau ",cls=Symbol,real=True)
######################numerical value
omega_c = 1.5
omega_m = 1.1
omega_p = 0.8
Delta_m = omega_m - omega_p
theta = 0.1  # radians
g0 = 0.2  # Small coupling
lmd=0.9*Delta_m

#####################################
# j,k,n1,n2=symbols("j,k,n1,n2",cls=Symbol)
j=4
k=4
n1=4
n2=4

################################################################derived quantities
# Derived parameters

mu = lmd * cos(theta) + Delta_m
beta = Delta_m - lmd * cos(theta)
Omega = sqrt(beta * mu)
D = lmd**2 * sin(theta)**2 + omega_p**2


x1,y2=symbols("x1,y2",cls=Symbol,real=True)
alpha=exp(lmd*sin(theta)*tau)
rho=omega_c*x1**2-Rational(1,2)

#Delta checked
Delta=-g0*sqrt(2/beta)*lmd*sin(theta)/D*rho*alpha \
      *sin(omega_p*tau)\
    +g0*sqrt(2/beta)*omega_p/D*rho*alpha \
      *cos(omega_p*tau)\
    -g0*sqrt(2/beta)*omega_p/D*rho


#delta checked
delta=-g0*sqrt(2/beta)*lmd*sin(theta)/D*alpha*sin(omega_p*tau)\
    +g0*sqrt(2/beta)*omega_p/D*alpha*cos(omega_p*tau)-g0*sqrt(2/beta)*omega_p/D

# rr=Delta/delta

# pprint(simplify(rr))
sigma=I
#arg1 checked
arg1=-sqrt(Omega)*alpha*Delta/sqrt(alpha**4-1)
#arg2 checked
arg2=-sigma*sqrt(Omega)*Delta/sqrt(alpha**4-1)

sum_R=0
min_k_n2=min(k,n2)
for R in range(0,min_k_n2+1):
    fact1=factorial(R)
    fact2=binomial(k,R)
    fact3=binomial(n2,R)
    fact4=(4*alpha/(sigma*(alpha**2-1)))**R
    sum_R+=fact1*fact2*fact3*fact4*hermite(k-R,arg1)*hermite(n2-R,arg2)

prefact1=1/sqrt(2**(k+n2-1)*factorial(k)*factorial(n2))
half=Rational(1,2)
prefact2=exp(-half*Omega*Delta**2/(1+alpha**2))

prefact3=(alpha**2-1)**((k+n2)/2)/(alpha**2+1)**((k+n2+1)/2)*sigma**n2

#I_kn2 checked
I_kn2=prefact1*prefact2*prefact3*sum_R
quarter=Rational(1,4)

#psi_j_c checked
psi_j_c=omega_c**quarter*1/sqrt(2**j*factorial(j)*sqrt(pi))\
        *exp(-half*omega_c*x1**2)*hermite(j,omega_c**half*x1)
#psi_n1_c checked
psi_n1_c=omega_c**quarter*1/sqrt(2**n1*factorial(n1)*sqrt(pi)) \
         *exp(-half*omega_c*x1**2)*hermite(n1,omega_c**half*x1)


g_simple_expression=psi_j_c*psi_n1_c*I_kn2

one_over_8=Rational(1,8)
g_expansion=0
for R in range(0,min(k,n2)+1):
    for m1 in range(0,j//2+1):
        for m2 in range(0,n1//2+1):
            for m3 in range(0,(k-R)//2+1):
                for m4 in range(0,(n2-R)//2+1):
                    for t in range(0,k+n2-2*R-2*m3-2*m4+1):
                        omegac_pow=Rational(j-2*m1+n1-2*m2+1,2)+t
                        fc1=omega_c**omegac_pow

                        Omega_pow=Rational(k+n2-2*R-2*m3-2*m4,2)
                        fc2=Omega**Omega_pow

                        delta_pow=k+n2-2*R-2*m3-2*m4
                        fc3=delta**delta_pow

                        fc4=(alpha**2-1)**(m3+m4)

                        fc5_pow=-k-n2-half+R+m3+m4
                        fc5=(alpha**2+1)**fc5_pow

                        two_pow=2*R+half*j-2*m1+half*n1-2*m2+t-half*k-half*n2+half
                        fc6=2**two_pow

                        fc7=(-1)**(n2+R+m1+m2+m3+t)

                        fc8=alpha**(k-2*m3)

                        fc9=sqrt(factorial(j)*factorial(n1)*factorial(k)*factorial(n2)/pi)

                        fc10=1/factorial(R)

                        fc11=1/(factorial(m1)*factorial(j-2*m1))

                        fc12=1/(factorial(m2)*factorial(n1-2*m2))

                        fc13=1/(factorial(m3)*factorial(k-R-2*m3))

                        fc14=1/(factorial(m4)*factorial(n2-R-2*m4))

                        fc15=factorial(k+n2-2*R-2*m3-2*m4)/(factorial(t)*factorial(k+n2-2*R-2*m3-2*m4-t))

                        exp_sum_part1=-half*Omega*delta**2/(1+alpha**2)*omega_c**2*x1**4
                        exp_sum_part2=omega_c*(half*Omega*delta**2/(1+alpha**2)-1)*x1**2
                        exp_sum_part3=-one_over_8*Omega*delta**2/(1+alpha**2)
                        fc16=exp(exp_sum_part1+exp_sum_part2+exp_sum_part3)

                        fc17=x1**(j-2*m1+n1-2*m2+2*t)

                        g_expansion+=fc1*fc2*fc3*fc4\
                                    *fc5*fc6*fc7*fc8\
                                    *fc9*fc10*fc11*fc12\
                                    *fc13*fc14*fc15*fc16*fc17



rst=g_simple_expression-g_expansion

rst=rst.subs([(x1,0.01),(y2,-0.2),(tau,0.01)]).evalf()
pprint(rst)