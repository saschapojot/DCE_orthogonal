//
// Created by adada on 12/5/2025.
//

#include "integral_table.hpp"


///
/// @param result pcfu
/// @param a
/// @param z
void table::pcfu(arb_t result, const arb_t a, const arb_t z)
{
    arb_t a_param, z_param;
    arb_t power_of_2, exp_term, z_sq, z_sq_half;
    arb_t temp;


    arb_init(a_param);

    arb_init(z_param);
    arb_init(power_of_2);
    arb_init(exp_term);
    arb_init(z_sq);
    arb_init(z_sq_half);
    arb_init(temp);

    // Compute z²
    arb_sqr(z_sq, z, prec);
    // Compute z²/2
    arb_mul_2exp_si(z_sq_half, z_sq, -1);

    // Compute a_param = 1/4 + a/2
    arb_mul_2exp_si(temp, a, -1);  // a/2
    arb_add(a_param, quarter, temp, prec);  // 1/4 + a/2
    // z_param = z²/2
    arb_set(z_param, z_sq_half);
    // Compute U(1/4 + a/2, 1/2, z²/2)
    arb_hypgeom_u(result, a_param, half, z_param, prec);

    // Compute 2^(-1/4 - a/2)
    arb_mul_2exp_si(temp, a, -1);  // a/2
    arb_add(temp, quarter, temp, prec);  // 1/4 + a/2
    arb_neg(temp, temp);  // -(1/4 + a/2)
    arb_pow(power_of_2, two, temp, prec);  // 2^(-1/4 - a/2)
    // Compute e^(-z²/4)
    arb_mul_2exp_si(exp_term, z_sq, -2);  // z²/4
    arb_neg(exp_term, exp_term);  // -z²/4
    arb_exp(exp_term, exp_term, prec);  // e^(-z²/4)

    // Multiply everything together
    // U(a, z) = 2^(-1/4 - a/2) * e^(-z²/4) * U(1/4 + a/2, 1/2, z²/2)
    arb_mul(result, result, power_of_2, prec);
    arb_mul(result, result, exp_term, prec);

    arb_clear(a_param);

    arb_clear(z_param);
    arb_clear(power_of_2);
    arb_clear(exp_term);
    arb_clear(z_sq);
    arb_clear(z_sq_half);
    arb_clear(temp);
}


///
/// @param result one term in summation of Z_tilde
/// @param j
/// @param k
/// @param n1
/// @param n2
/// @param R
/// @param m1
/// @param m2
/// @param m3
/// @param m4
/// @param t
void table::summation_one_term(arb_t result,int j,int k,int n1,int n2,int R, int m1, int m2, int m3, int m4,int t)
{

    arb_t temp, temp2, temp3, temp4;
    arb_t factor1, factor2, factor3, factor4, factor5, factor6, factor7, factor8;
    arb_t alpha_sq, alpha_sq_plus_1, alpha_sq_minus_1;
    arb_t omega_delta_sq, exp_arg, exp_term;
    arb_t gamma_arg, gamma_val;
    arb_t U_a_param, U_z_param;
    arb_t sqrt_term, factorial_product;
    arb_t power_term;

    // Initialize all temporary variables
    arb_init(temp); arb_init(temp2); arb_init(temp3); arb_init(temp4);
    arb_init(factor1); arb_init(factor2); arb_init(factor3); arb_init(factor4);
    arb_init(factor5); arb_init(factor6); arb_init(factor7); arb_init(factor8);
    arb_init(alpha_sq); arb_init(alpha_sq_plus_1); arb_init(alpha_sq_minus_1);
    arb_init(omega_delta_sq); arb_init(exp_arg); arb_init(exp_term);
    arb_init(gamma_arg); arb_init(gamma_val);
    arb_init(U_a_param); arb_init(U_z_param);
    arb_init(sqrt_term); arb_init(factorial_product);
    arb_init(power_term);

    // Precompute commonly used terms
    arb_sqr(alpha_sq, this->alpha, prec);                      // α²
    arb_add_ui(alpha_sq_plus_1, alpha_sq, 1, prec);           // α² + 1
    arb_sub_ui(alpha_sq_minus_1, alpha_sq, 1, prec);          // α² - 1
    arb_sqr(temp, this->delta, prec);
    arb_mul(omega_delta_sq, this->Omega, temp, prec);         // Ω·δ²

    // --- Factor 1: ω_c^((j-2m1+n1-2m2+1)/2 + t) ---
    int exp1 = j - 2*m1 + n1 - 2*m2 + 1;
    arb_set_si(temp, exp1);
    arb_mul_2exp_si(temp, temp, -1);  // (j-2m1+n1-2m2+1)/2
    arb_add_si(temp, temp, t, prec);  // + t
    arb_pow(factor1, this->omegac, temp, prec);
    // --- Factor 2: Ω^((k+n2-2R-2m3-2m4)/2) ---
    int exp2 = k + n2 - 2*R - 2*m3 - 2*m4;
    arb_set_si(temp, exp2);
    arb_mul_2exp_si(temp, temp, -1);  // /2
    arb_pow(factor2, this->Omega, temp, prec);

    // --- Factor 3: δ^(k+n2-2R-2m3-2m4) ---
    arb_pow_ui(factor3, this->delta, exp2, prec);

    // --- Factor 4: (α²-1)^(m3+m4) ---
    arb_pow_ui(factor4, alpha_sq_minus_1, m3 + m4, prec);
    // --- Factor 5: (α²+1)^(-k-n2-1/2+R+m3+m4) ---
    arb_set_si(temp, -k - n2 + R + m3 + m4);
    arb_sub(temp, temp, half, prec);  // -k-n2-1/2+R+m3+m4
    arb_pow(factor5, alpha_sq_plus_1, temp, prec);

    // --- Factor 6: 2^(2R + j/2 - 2m1 + n1/2 - 2m2 + t - k/2 - n2/2 + 1/2) ---
    arb_set_si(temp, 2*R - 2*m1 - 2*m2 + t);
    arb_set_si(temp2, j + n1 - k - n2+1);
    arb_mul_2exp_si(temp2, temp2, -1);  // (j+n1-k-n2)/2
    arb_add(temp, temp, temp2, prec);
    arb_pow(factor6, this->two, temp, prec);

    // --- Factor 7: (-1)^(n2+R+m1+m2+m3+t) ---
    int sign_exp = n2 + R + m1 + m2 + m3 + t;
    arb_set_si(factor7, (sign_exp % 2 == 0) ? 1 : -1);

    // --- Factor 8: α^(k-2m3) ---
    arb_pow_ui(factor8, this->alpha, k - 2*m3, prec);

    // --- Square root term: sqrt(j!·n1!·k!·n2!/π) ---
    arb_fac_ui(temp, j, prec);
    arb_fac_ui(temp2, n1, prec);
    arb_mul(temp3, temp, temp2, prec);
    arb_fac_ui(temp, k, prec);
    arb_mul(temp3, temp3, temp, prec);
    arb_fac_ui(temp, n2, prec);
    arb_mul(temp3, temp3, temp, prec);
    arb_div(temp3, temp3, this->pi, prec);
    arb_sqrt(sqrt_term, temp3, prec);

    // --- Factorial product in denominator ---
    // 1/(R! · m1!(j-2m1)! · m2!(n1-2m2)! · m3!(k-R-2m3)! · m4!(n2-R-2m4)! · [t!(k+n2-2R-2m3-2m4-t)!/(k+n2-2R-2m3-2m4)!])
    arb_fac_ui(factorial_product, R, prec);

    arb_fac_ui(temp, m1, prec);
    arb_fac_ui(temp2, j - 2*m1, prec);
    arb_mul(temp, temp, temp2, prec);
    arb_mul(factorial_product, factorial_product, temp, prec);


    arb_fac_ui(temp, m2, prec);
    arb_fac_ui(temp2, n1 - 2*m2, prec);
    arb_mul(temp, temp, temp2, prec);
    arb_mul(factorial_product, factorial_product, temp, prec);

    arb_fac_ui(temp, m3, prec);
    arb_fac_ui(temp2, k - R - 2*m3, prec);
    arb_mul(temp, temp, temp2, prec);
    arb_mul(factorial_product, factorial_product, temp, prec);

    arb_fac_ui(temp, m4, prec);
    arb_fac_ui(temp2, n2 - R - 2*m4, prec);
    arb_mul(temp, temp, temp2, prec);
    arb_mul(factorial_product, factorial_product, temp, prec);

    // (k+n2-2R-2m3-2m4)! / [t! · (k+n2-2R-2m3-2m4-t)!]
    arb_fac_ui(temp, exp2, prec);
    arb_fac_ui(temp2, t, prec);
    arb_mul(factorial_product, factorial_product, temp2, prec);
    arb_fac_ui(temp2, exp2 - t, prec);
    arb_mul(factorial_product, factorial_product, temp2, prec);
    arb_div(factorial_product, temp, factorial_product, prec);

    // --- Exponential term: exp[1/4 · (1+α²)/(Ω·δ²) · (1/2·Ω·δ²/(1+α²) - 1)²] ---
    // Compute inner term: (1/2·Ω·δ²/(1+α²) - 1)
    arb_div(temp, omega_delta_sq, alpha_sq_plus_1, prec);
    arb_mul_2exp_si(temp, temp, -1);  // 1/2 · Ω·δ²/(1+α²)
    arb_sub_ui(temp, temp, 1, prec);  // - 1
    arb_sqr(temp, temp, prec);        // square it

    // Multiply by (1+α²)/(Ω·δ²)
    arb_div(temp2, alpha_sq_plus_1, omega_delta_sq, prec);
    arb_mul(temp, temp, temp2, prec);
    // Multiply by 1/4
    arb_mul_2exp_si(temp, temp, -2);  // first term complete
    // Second term: -1/8·Ω·δ²/(1+α²)
    arb_div(temp2, omega_delta_sq, alpha_sq_plus_1, prec);
    arb_mul_2exp_si(temp2, temp2, -3);  // 1/8 · Ω·δ²/(1+α²)
    arb_neg(temp2, temp2);              // negate to get -1/8·Ω·δ²/(1+α²)

    // Add both terms
    arb_add(exp_arg, temp, temp2, prec);
    arb_exp(exp_term, exp_arg, prec);

    // --- Power term: [√((1+α²)/Ω)·1/(|δ|·ωc)]^((j-2m₁+n₁-2m₂+2t+1)/2) ---
    int gamma_exp = j - 2*m1 + n1 - 2*m2 + 2*t + 1;
    arb_set_si(temp2, gamma_exp);
    arb_mul_2exp_si(temp2, temp2, -1);  // (j-2m₁+n₁-2m₂+2t+1)/2
    // Compute the base: √((1+α²)/Ω) · 1/(|δ|·ωc)
    arb_div(temp, alpha_sq_plus_1, this->Omega, prec);
    arb_sqrt(temp, temp, prec);  // √((1+α²)/Ω)

    arb_abs(temp3, this->delta);        // |δ|
    arb_mul(temp3, temp3, this->omegac, prec);  // |δ|·ωc
    arb_div(temp, temp, temp3, prec);   // √((1+α²)/Ω) / (|δ|·ωc)
    // Raise to the power
    arb_pow(power_term, temp, temp2, prec);   // [...]^((j-2m₁+n₁-2m₂+2t+1)/2)

    // --- Gamma function: Γ((j-2m₁+n₁-2m₂+2t+1)/2) ---
    // Note: temp2 already contains (j-2m₁+n₁-2m₂+2t+1)/2
    arb_gamma(gamma_val, temp2, prec);
    // --- U function: U(a, z) where ---
    // a = (j-2m₁+n₁-2m₂+2t)/2
    // z = -√((1+α²)/Ω) · (1/2·Ω·δ²/(1+α²) - 1) / |δ|
    // Compute a parameter
    arb_set_si(U_a_param, j - 2*m1 + n1 - 2*m2 + 2*t);
    arb_mul_2exp_si(U_a_param, U_a_param, -1);

    // Compute z parameter
    arb_div(temp, alpha_sq_plus_1, this->Omega, prec);
    arb_sqrt(temp, temp, prec);  // √((1+α²)/Ω)

    arb_div(temp2, omega_delta_sq, alpha_sq_plus_1, prec);
    arb_mul_2exp_si(temp2, temp2, -1);  // 1/2·Ω·δ²/(1+α²)
    arb_sub_ui(temp2, temp2, 1, prec);  // - 1

    arb_mul(temp, temp, temp2, prec);   // multiply
    arb_abs(temp2, this->delta);        // |δ|
    arb_div(temp, temp, temp2, prec);   // divide by |δ|
    arb_neg(U_z_param, temp);           // negate

    // Call pcfu function
    arb_t U_result;
    arb_init(U_result);
    pcfu(U_result, U_a_param, U_z_param);

    // --- Multiply all factors together ---
    arb_mul(result, factor1, factor2, prec);
    arb_mul(result, result, factor3, prec);
    arb_mul(result, result, factor4, prec);
    arb_mul(result, result, factor5, prec);
    arb_mul(result, result, factor6, prec);
    arb_mul(result, result, factor7, prec);
    arb_mul(result, result, factor8, prec);
    arb_mul(result, result, sqrt_term, prec);
    arb_mul(result, result, factorial_product, prec);
    arb_mul(result, result, exp_term, prec);
    arb_mul(result, result, power_term, prec);
    arb_mul(result, result, gamma_val, prec);
    arb_mul(result, result, U_result, prec);


    // Clear all temporary variables
    arb_clear(temp); arb_clear(temp2); arb_clear(temp3); arb_clear(temp4);
    arb_clear(factor1); arb_clear(factor2); arb_clear(factor3); arb_clear(factor4);
    arb_clear(factor5); arb_clear(factor6); arb_clear(factor7); arb_clear(factor8);
    arb_clear(alpha_sq); arb_clear(alpha_sq_plus_1); arb_clear(alpha_sq_minus_1);
    arb_clear(omega_delta_sq); arb_clear(exp_arg); arb_clear(exp_term);
    arb_clear(gamma_arg); arb_clear(gamma_val);
    arb_clear(U_a_param); arb_clear(U_z_param);
    arb_clear(sqrt_term); arb_clear(factorial_product);
    arb_clear(power_term); arb_clear(U_result);

}



///
/// @brief Set alpha = e^(lambda*sin(theta)*tau)
/// @param tau time parameter
void table::set_alpha(const arb_t tau)
{
    arb_t temp, sin_theta;
    arb_init(temp);
    arb_init(sin_theta);
    // Compute sin(theta)
    arb_sin(sin_theta, this->theta, prec);
    // Compute lambda * sin(theta) * tau
    arb_mul(temp, this->lmd, sin_theta, prec);
    arb_mul(temp, temp, tau, prec);
    // Compute e^(lambda*sin(theta)*tau) and store in this->alpha
    arb_exp(this->alpha, temp, prec);

    arb_clear(temp);
    arb_clear(sin_theta);

}


///
/// @brief Set delta using current alpha value
/// @param tau time parameter
void table::set_delta(const arb_t tau)
{
    arb_t temp1, temp2, temp3;
    arb_t sin_theta, sin_omegap_tau, cos_omegap_tau;
    arb_t sqrt_2_over_beta, g0_sqrt_factor;
    arb_t term1, term2, term3;

    arb_init(temp1); arb_init(temp2); arb_init(temp3);
    arb_init(sin_theta); arb_init(sin_omegap_tau); arb_init(cos_omegap_tau);
    arb_init(sqrt_2_over_beta); arb_init(g0_sqrt_factor);
    arb_init(term1); arb_init(term2); arb_init(term3);

    // Compute sin(theta)
    arb_sin(sin_theta, this->theta, prec);
    // Compute sin(omega_p * tau) and cos(omega_p * tau)
    arb_mul(temp1, this->omegap, tau, prec);
    arb_sin_cos(sin_omegap_tau, cos_omegap_tau, temp1, prec);
    // Compute sqrt(2/beta)
    arb_div(temp1, this->two, this->beta, prec);
    arb_sqrt(sqrt_2_over_beta, temp1, prec);
    // Compute g0 * sqrt(2/beta)
    arb_mul(g0_sqrt_factor, this->g0, sqrt_2_over_beta, prec);

    // --- Compute first term: -g0*sqrt(2/beta) * (lambda*sin(theta)/D) * alpha * sin(omega_p*tau) ---
    arb_mul(temp1, this->lmd, sin_theta, prec);  // lambda * sin(theta)
    arb_div(temp1, temp1, this->D, prec);         // (lambda * sin(theta)) / D
    arb_mul(temp1, temp1, this->alpha, prec);     // * alpha
    arb_mul(temp1, temp1, sin_omegap_tau, prec);  // * sin(omega_p * tau)
    arb_mul(term1, g0_sqrt_factor, temp1, prec);  // * g0*sqrt(2/beta)
    arb_neg(term1, term1);                        // negate

    // --- Compute second term: g0*sqrt(2/beta) * (omega_p/D) * alpha * cos(omega_p*tau) ---
    arb_div(temp2, this->omegap, this->D, prec);  // omega_p / D
    arb_mul(temp2, temp2, this->alpha, prec);     // * alpha
    arb_mul(temp2, temp2, cos_omegap_tau, prec);  // * cos(omega_p * tau)
    arb_mul(term2, g0_sqrt_factor, temp2, prec);  // * g0*sqrt(2/beta)

    // --- Compute third term: -g0*sqrt(2/beta) * (omega_p/D) ---
    arb_div(temp3, this->omegap, this->D, prec);  // omega_p / D
    arb_mul(term3, g0_sqrt_factor, temp3, prec);  // * g0*sqrt(2/beta)
    arb_neg(term3, term3);                        // negate

    // Sum all three terms
    arb_add(this->delta, term1, term2, prec);
    arb_add(this->delta, this->delta, term3, prec);

    // Clear all temporary variables
    arb_clear(temp1); arb_clear(temp2); arb_clear(temp3);
    arb_clear(sin_theta); arb_clear(sin_omegap_tau); arb_clear(cos_omegap_tau);
    arb_clear(sqrt_2_over_beta); arb_clear(g0_sqrt_factor);
    arb_clear(term1); arb_clear(term2); arb_clear(term3);


}