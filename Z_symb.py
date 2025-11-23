from sympy import *

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
# Derived parameters
lmd=0.9*Delta_m
j,k,n1,n2=symbols("j,k,n1,n2",cls=Symbol)

#####################################
j=3
k=3
n1=3
n2=3

################################################################derived quantities
mu = lmd * cos(theta) + Delta_m
beta = Delta_m - lmd * cos(theta)
Omega = sqrt(beta * mu)
D = lmd**2 * sin(theta)**2 + omega_p**2


x1,y2=symbols("x1,y2",cls=Symbol,real=True)
alpha=exp(lmd*sin(theta)*tau)
rho=omega_c*x1**2-Rational(1,2)

Delta=-g0*sqrt(2/beta)*lmd*sin(theta)/D*rho*alpha \
      *sin(omega_p*tau)\
    +g0*sqrt(2/beta)*omega_p/D*rho*alpha \
      *cos(omega_p*tau)\
    -g0*sqrt(2/beta)*omega_p/D*rho



delta=-g0*sqrt(2/beta)*lmd*sin(theta)/D*alpha*sin(omega_p*tau)\
    +g0*sqrt(2/beta)*omega_p/D*alpha*cos(omega_p*tau)-g0*sqrt(2/beta)*omega_p/D

sigma=I
arg1=-sqrt(Omega)*alpha*Delta/sqrt(alpha**4-1)
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


I_kn2=prefact1*prefact2*prefact3*sum_R
quarter=Rational(1,4)
psi_j_c=omega_c**quarter*1/sqrt(2**j*factorial(j)*sqrt(pi))\
        *exp(-half*omega_c*x1**2)*hermite(j,omega_c**half*x1)
psi_n1_c=omega_c**quarter*1/sqrt(2**n1*factorial(n1)*sqrt(pi)) \
         *exp(-half*omega_c*x1**2)*hermite(n1,omega_c**half*x1)

gx1_explicit_part1=omega_c**half*1/sqrt(2**(j+n1+k+n2-1)*factorial(j)*factorial(n1)*factorial(k)*factorial(n2)*pi)\
            *(alpha**2-1)**((k+n2)/2)/(alpha**2+1)**((k+n2+1)/2)*sigma**n2\
            *exp(-omega_c*x1**2)*exp(-half*Omega*Delta**2/(1+alpha**2))

gx1_explicit_part2=0
for R in range(0,min_k_n2+1):
    fact1=factorial(R)
    fact2=binomial(k,R)
    fact3=binomial(n2,R)
    fact4=(4*alpha/(sigma*(alpha**2-1)))**R
    H_arg1=omega_c**half*x1
    H_arg2=omega_c**half*x1
    H_arg3=-sqrt(Omega)*alpha*Delta/sqrt(alpha**4-1)
    H_arg4=-sigma*sqrt(Omega)*Delta/sqrt(alpha**4-1)

    gx1_explicit_part2+=fact1*fact2*fact3*fact4*hermite(j,H_arg1)\
                        *hermite(n1,H_arg2)*hermite(k-R,H_arg3)\
                        *hermite(n2-R,H_arg4)

gx_explicit=gx1_explicit_part1*gx1_explicit_part2





def full_H(n,x):
    val=0
    for m in range(0,n//2+1):
        val+=(-1)**m/(factorial(m)*factorial(n-2*m))*(2*x)**(n-2*m)

    val*=factorial(n)
    return val


arg1=omega_c**half*x1
Hj_func=full_H(j,arg1)

arg2=omega_c**half*x1
Hn1_func=full_H(n1,arg2)

R=1
arg3=-sqrt(Omega)*alpha*Delta/sqrt(alpha**4-1)
H_kmR=full_H(k-R,arg3)

arg4=-sigma*sqrt(Omega)*Delta/sqrt(alpha**4-1)
H_n2mR=full_H(n2-R,arg4)

four_H_prod_expand=0


for m1 in range(0,j//2+1):
    for m2 in range(0,n1//2+1):
        for m3 in range(0,(k-R)//2+1):
            for m4 in range(0,(n2-R)//2+1):
                part1=(-1)**(m1+m2+m3+m4+k+n2)
                pow_2=j-2*m1+n1-2*m2+k-2*R-2*m3+n2-2*m4
                part2=2**pow_2

                part3=1/(factorial(m1)*factorial(j-2*m1))\
                     * 1/(factorial(m2)*factorial(n2-2*m2))\
                    * 1/(factorial(m3)*factorial(k-R-2*m3))\
                    * 1/(factorial(m4)*factorial(n2-R-2*m4))
                pow_omega_c=(j-2*m1+n1-2*m2)/2
                part4=omega_c**pow_omega_c
                pow_Omega=(k+n2-2*R-2*m3-2*m4)/2
                part5=Omega**pow_Omega
                part6=sigma**(n2-R-2*m4)
                part7=alpha**(k-R-2*m3)
                part8=(1/sqrt(alpha**4-1))**(k+n2-2*R-2*m3-2*m4)

                pow_x1=j-2*m1+n1-2*m2+k-2*R-2*m3+n2-2*m4
                part9=x1**pow_x1

                part10=Delta**(k+n2-2*R-2*m3-2*m4)

                four_H_prod_expand+=part1*part2*part3*part4\
                                    *part5*part6*part7*part8\
                                    *part9*part10




four_H_prod_expand*=factorial(j)*factorial(n1)*factorial(k-R)*factorial(n2-R)

direct_4H_prod=Hj_func*Hn1_func*H_kmR*H_n2mR

rst=four_H_prod_expand-direct_4H_prod
rst=rst.subs([(x1,0.1),(y2,0.05),(tau,0.2)]).evalf()
pprint(simplify(expand(rst)))