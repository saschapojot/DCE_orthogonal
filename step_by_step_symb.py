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
x2,dx2=symbols("x2,dx2",cls=Symbol,real=True)

half=Rational(1,2)
quarter=Rational(1,4)
rho=omegac*x1**2-half

D=lmd**2*sin(theta)**2+omegap**2
mu=lmd*cos(theta)+Deltam

