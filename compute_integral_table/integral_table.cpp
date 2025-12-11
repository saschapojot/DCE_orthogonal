//
// Created by adada on 12/5/2025.
//

#include "integral_table.hpp"

///
/// @param nu
/// @param x
/// @return Function to calculate D_nu(x) using GSL's Hypergeometric U
double table::pcf_D(const double& nu, const double&  x)
{
    // Formula: D_v(x) = 2^(v/2) * exp(-x^2/4) * U(-v/2, 1/2, x^2/2)

    double term1 = pow(2.0, nu / 2.0);
    double term2 = exp(- (x * x) / 4.0);

    // gsl_sf_hyperg_U(a, b, z)
    // Here a = -nu/2, b = 1/2, z = x^2/2
    double term3 = gsl_sf_hyperg_U(-nu / 2.0, 0.5, (x * x) / 2.0);

    return term1 * term2 * term3;
}


///
/// @param a
/// @param z
/// @return Parabolic Cylinder Function U(a,z)
double table::pcf_U(const double& a, const double& z)
{
    double nu=-0.5-a;
    return this->pcf_D(nu,z);
}
