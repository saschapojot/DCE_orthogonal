//
// Created by adada on 11/6/2025.
//

#include "evolution.hpp"
///for testing
void evolution::print_anything()
{
    //print anything
    int n=200;
    double x=100;
    // std::cout<<hermite_polynomial(n,x)<<std::endl;
    std::cout<<harmonic_oscillator_wavefunction(n,x,1.3)<<std::endl;
}

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

///
/// @param n order
/// @param x variable
/// @return hermite polynomial value h_n(x)
double evolution::hermite_polynomial(int n, double x)
{
    return boost::math::hermite(n, x);
}


///
/// @param n order of Hermite function
/// @param x variable
/// @param omega frequency
/// @return Hermite function (normalized)
double evolution::harmonic_oscillator_wavefunction(int n, double x, double omega)
{

    const double sqrt_pi = std::sqrt(PI);
    // Compute normalization factor in log space to avoid overflow
    double log_normalization = -0.5 * (static_cast<double>(n) * std::log(2.0) + std::lgamma(n + 1) + std::log(sqrt_pi));
    double scaled_x = std::sqrt(omega) * x;
    double hermite_val = hermite_polynomial(n, scaled_x);
    double exp_val = std::exp(-omega * x * x / 2.0);
    // Combine everything carefully
    return std::pow(omega, 0.25) * std::exp(log_normalization) * hermite_val * exp_val;
}


