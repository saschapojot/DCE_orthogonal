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


c0_integral=I*g0**2/D*omegap*rho**2*tau\
    +I*g0**2/(2*D)*rho**2*(lmd*sin(theta)/omegap*cos(2*omegap*tau)+sin(2*omegap*tau))\
    -I*g0/D*sqrt(2*beta)*rho*(s2+g0/D*sqrt(2/beta)*omegap*rho)*exp(-lmd*sin(theta)*tau)\
    *(omegap*sin(omegap*tau)-lmd*sin(theta)*cos(omegap*tau))

c0_int_tau0=I*g0/D*sqrt(2*beta)*lmd*sin(theta)*rho*s2\
    +I*g0**2*lmd*sin(theta)*(D+4*omegap**2)/(2*omegap*D**2)*rho**2

s2_in_y2=y2*exp(lmd*sin(theta)*tau)-g0*sqrt(2/beta)*lmd*sin(theta)/D*rho*exp(lmd*sin(theta)*tau)*sin(omegap*tau)\
    +g0*sqrt(2/beta)*omegap/D*rho*cos(omegap*tau)*exp(lmd*sin(theta)*tau)\
    -g0*sqrt(2/beta)*omegap/D*rho

G_lhs=c0_integral-c0_int_tau0

G_rhs=I*g0**2/D*omegap*rho**2*tau\
    +I*g0**2/(2*D)*lmd*sin(theta)/omegap*rho**2*cos(2*omegap*tau)\
    +I*g0**2/(2*D)*rho**2*sin(2*omegap*tau)\
    -I*g0/D*omegap*rho*(sqrt(2*beta)*s2+2*g0/D*omegap*rho)*exp(-lmd*sin(theta)*tau)*sin(omegap*tau)\
    +I*g0/D*lmd*sin(theta)*rho*(sqrt(2*beta)*s2+2*g0/D*omegap*rho)*exp(-lmd*sin(theta)*tau)*cos(omegap*tau)\
    -I*g0/D*sqrt(2*beta)*lmd*sin(theta)*rho*s2-I*g0**2*lmd*sin(theta)*(D+4*omegap**2)/(2*omegap*D**2)*rho**2

F0=I*g0**2/D**2*lmd*sin(theta)*rho**2*(4*omegap**2-D)/(2*omegap)

F1=I*g0**2/D*omegap*rho**2

F2=-I*g0/D*omegap*sqrt(2*beta)*rho*y2

F3=I*g0/D*lmd*sin(theta)*sqrt(2*beta)*rho*y2

F4=-I*g0**2/(2*D)*rho**2

F5=I*g0**2/(2*D)*lmd*sin(theta)/omegap*rho**2

F6=-I*g0/D*sqrt(2*beta)*lmd*sin(theta)*rho*y2

F7=I*2*g0**2/D**2*lmd**2*(sin(theta))**2*rho**2

F8=-I*2*g0**2/D**2*omegap*lmd*sin(theta)*rho**2

G=F0+F1*tau+F2*sin(omegap*tau)+F3*cos(omegap*tau)+F4*sin(2*omegap*tau)\
    +F5*cos(2*omegap*tau)+F6*exp(lmd*sin(theta)*tau)\
    +F7*exp(lmd*sin(theta)*tau)*sin(omegap*tau)+F8*exp(lmd*sin(theta)*tau)*cos(omegap*tau)

rst=G_rhs.subs([(s2,s2_in_y2)])-G


tmp=TR5(TR11(rst))

pprint(simplify(expand(tmp)))


