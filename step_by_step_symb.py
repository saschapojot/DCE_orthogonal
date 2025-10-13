from sympy import *
from sympy import expand_complex
from sympy.simplify.fu import TR11,TR5
import pandas as pd
import numpy as np

#symbolic computation for orthogonal notes


g0,lmd,theta=symbols("g0,lambda,theta",cls=Symbol,real=True)
x1,tau,s2,t=symbols("x1,tau,s2,t",cls=Symbol,real=True)
Deltam=symbols("Delta_m",cls=Symbol,real=True)
omegam,omegac,omegap=symbols("omega_m,omega_c,omega_p",cls=Symbol,positive=True)
y2,dy2=symbols("y2,dy2",cls=Symbol,real=True)
beta=symbols("beta",cls=Symbol,positive=True)
half=Rational(1,2)
quarter=Rational(1,4)
rho=omegac*x1**2-half

D=lmd**2*sin(theta)**2+omegap**2
mu=lmd*cos(theta)+Deltam

y2_solution=g0*sqrt(2/beta)*rho*(lmd*sin(theta)*sin(omegap*tau)-omegap*cos(omegap*tau))/D\
    +(s2+g0*sqrt(2/beta)*rho*omegap/D)*exp(-lmd*sin(theta)*tau)

c2_in_y2_solution=g0*sqrt(2/beta)*rho*sin(omegap*tau)-lmd*sin(theta)*y2_solution



y2_in_exp=g0*sqrt(2/beta)*rho*(-I*half*lmd*sin(theta)-half*omegap)/D*exp(I*omegap*tau)\
    +g0*sqrt(2/beta)*rho*(I*half*lmd*sin(theta)-half*omegap)/D*exp(-I*omegap*tau)\
    +(s2+g0*sqrt(2/beta)*rho*omegap/D)*exp(-lmd*sin(theta)*tau)

c0_in_y2_solution=-I*g0*sqrt(2*beta)*rho*y2_solution*cos(omegap*tau)



c0_exp=I*g0**2/D*omegap*rho**2\
    +g0**2/D*rho**2*(-half*lmd*sin(theta)+I*half*omegap)*exp(I*2*omegap*tau)\
    +g0**2/D*rho**2*(half*lmd*sin(theta)+I*half*omegap)*exp(-I*2*omegap*tau)\
    -I*half*g0*sqrt(2*beta)*rho*(s2+g0/D*sqrt(2/beta)*omegap*rho)*exp((-lmd*sin(theta)+I*omegap)*tau)\
    -I*half*g0*sqrt(2*beta)*(s2+g0/D*sqrt(2/beta)*omegap*rho)*exp((-lmd*sin(theta)-I*omegap)*tau)

rst=c0_in_y2_solution-c0_exp

tmp=TR11(expand_complex(rst))

pprint(simplify(expand(tmp)))