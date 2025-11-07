//
// Created by adada on 11/6/2025.
//

#include "evolution.hpp"


///
/// @param tau time step
/// @return auxiliary function alpha for computing tensor Z
double evolution::alpha(double tau)
{
    return std::exp(lmd*std::sin(theta)*tau);
}

///
/// @param x1 variable for photon harmonic oscillator
/// @return auxiliary function
double evolution::rho(double x1)
{
    return omegac*std::pow(x1,2.0)-0.5;
}


///
/// @param x1 variable for photon harmonic oscillator
/// @param tau time step
/// @return auxiliary function Delta for computing tensor Z
double evolution::Delta(double x1, double tau)
{
    double part1=-g0*std::sqrt(2.0/beta)*lmd*std::sin(theta)/D*rho(x1)*alpha(tau)*std::sin(omegap*tau);

    double part2=g0*std::sqrt(2.0/beta)*omegap/D*rho(x1)*std::cos(omegap*tau)*alpha(tau);
    double part3=-g0*std::sqrt(2.0/beta)*omegap/D*rho(x1);

    return part1+part2+part3;
}