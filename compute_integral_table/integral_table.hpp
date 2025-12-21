//
// Created by adada on 12/5/2025.
//

#ifndef INTEGRAL_TABLE_HPP
#define INTEGRAL_TABLE_HPP
#include <algorithm>
#include <arb.h>
#include <arb_hypgeom.h>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cmath>
#include <complex>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <flint.h>
#include <fstream>

#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>


namespace fs = boost::filesystem;
namespace bp = boost::python;
namespace np = boost::python::numpy;





class table
{
public:
    // Disable copy constructor and assignment to avoid double-free issues with arb_t
    table(const table&) = delete;
    table& operator=(const table&) = delete;

    table(const std::string &cppInParamsFileName)
    {
        // Initialize all arb_t variables
        arb_init(pi);
        arb_init(half);
        arb_init(quarter);
        arb_init(two);


        arb_init(g0);
        arb_init(omegam);
        arb_init(omegap);
        arb_init(omegac);
        arb_init(er);
        arb_init(thetaCoef);
        arb_init(theta);
        arb_init(lmd);
        arb_init(Deltam);
        arb_init(r);
        arb_init(e2r);
        arb_init(tTot);
        arb_init(dt);
        arb_init(D);
        arb_init(mu);
        arb_init(beta);
        arb_init(Omega);

        arb_init(alpha);
        arb_init(delta);

        // Initialize  constants
        arb_const_pi(pi, prec);
        arb_set_ui(quarter, 1);
        arb_mul_2exp_si(quarter, quarter, -2); // 1 * 2^-2 = 0.25

        arb_set_ui(half,1);
        arb_mul_2exp_si(half, half, -1);
        arb_set_ui(two, 2);


        std::ifstream file(cppInParamsFileName);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0;
        std::string tempStr; // Buffer for reading numbers
        while (std::getline(file, line))
        {
            // Check if the line is empty
            if (line.empty()) {
                continue; // Skip empty lines
            }
            std::istringstream iss(line);
            //read j1H
            if (paramCounter == 0)
            {
                iss>>j1H;
                if (j1H<0)
                {
                    std::cerr << "j1H must be >=0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }//end reading j1H

            //read j2H
            if (paramCounter == 1)
            {
                iss>>j2H;
                if (j2H<0)
                {
                    std::cerr << "j2H must be >=0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;

            }//end reading j2H

            //read g0
            if (paramCounter == 2)
            {
                iss >> tempStr;
                arb_set_str(g0, tempStr.c_str(), prec);
                paramCounter++;
                continue;
            }//end reading g0

            //read omegam
            if(paramCounter == 3)
            {
                iss >> tempStr; arb_set_str(omegam, tempStr.c_str(), prec);
                paramCounter++;
                continue;
            }//end reading omegam

            //read omegap
            if(paramCounter == 4)
            {
                iss >> tempStr;
                arb_set_str(omegap, tempStr.c_str(), prec);
                paramCounter++;
                continue;
            }
            //end reading omegap
            //read omegac
            if(paramCounter == 5)
            {
                iss >> tempStr;
                arb_set_str(omegac, tempStr.c_str(), prec);
                paramCounter++;
                continue;
            }//end reading omegac

            //read er
            if(paramCounter == 6)
            {
                iss >> tempStr;
                arb_set_str(er, tempStr.c_str(), prec);
                if(!arb_is_positive(er)) { std::cerr << "er must be >0" << std::endl; std::exit(1); }
                paramCounter++;
                continue;
            }//end reading er

            //read thetaCoef
            if(paramCounter == 7)
            {
                iss >> tempStr;
                arb_set_str(thetaCoef, tempStr.c_str(), prec);
                paramCounter++;
                continue;
            }//end reading thetaCoef

            //read groupNum
            if (paramCounter==8)
            {
                iss>>groupNum;
                paramCounter++;
                continue;
            }//end groupNum

            //read rowNum
            if (paramCounter==9)
            {
                iss>>rowNum;
                paramCounter++;
                continue;
            }//end rowNum

            //read N1
            if (paramCounter==10)
            {
                iss>>N1;
                if (N1<=0)
                {
                    std::cerr << "N1 must be >0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }// end N1

            //read N2
            if (paramCounter==11)
            {
                iss>>N2;
                if (N2<=0)
                {
                    std::cerr << "N2 must be >0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }//end N2

            //read tTot
            if (paramCounter==12)
            {
                iss >> tempStr;
                arb_set_str(tTot, tempStr.c_str(), prec);
                if(!arb_is_positive(tTot)) { std::cerr << "tTot must be >0" << std::endl; std::exit(1); }
                paramCounter++;
                continue;
            }//end tTot

            //read Q
            if (paramCounter==13)
            {
                iss>>Q;
                if (Q<=0)
                {
                    std::cerr << "Q must be >0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }//end Q

        }//end while
        //print parameters
        std::cout << std::setprecision(15);
        std::cout << "j1H=" << j1H << ", j2H=" << j2H << ", ";
        print_arb("g0", g0); std::cout << ", ";
        print_arb("omegam", omegam); std::cout << ", ";
        print_arb("omegap", omegap); std::cout << ", ";
        print_arb("omegac", omegac); std::cout << ", ";
        print_arb("er", er); std::cout << ", ";
        print_arb("thetaCoef", thetaCoef); std::cout << ", ";
        std::cout << "groupNum=" << groupNum << ", rowNum=" << rowNum
                  << ", N1=" << N1 << ", N2=" << N2 << ", ";
        print_arb("tTot", tTot); std::cout << ", Q=" << Q << std::endl;


        // --- Compute derived quantities ---
        // Temporary variables for calculation
        arb_t tmp1, tmp2, tmp3, tmp_sin, tmp_cos;
        arb_init(tmp1); arb_init(tmp2);
        arb_init(tmp3); arb_init(tmp_sin); arb_init(tmp_cos);

        // 1. Calculate theta = thetaCoef * PI
        arb_mul(this->theta, this->thetaCoef, this->pi, prec);

        // 2. Calculate r = log(er)
        arb_log(this->r, this->er, prec);

        // 3. Calculate e^2r = er * er
        arb_mul(this->e2r, this->er, this->er, prec);

        // 4. Calculate time step dt = tTot / Q
        arb_set_si(tmp1, this->Q); // Convert int Q to arb
        arb_div(this->dt, this->tTot, tmp1, prec);

        // 5. Deltam = omegam - omegap
        arb_sub(this->Deltam, this->omegam, this->omegap, prec);

        // 6. lmd = (e2r - 1/e2r)/(e2r + 1/e2r) * Deltam
        // tmp1 = 1/e2r
        arb_inv(tmp1, this->e2r, prec);
        // tmp2 = e2r - 1/e2r
        arb_sub(tmp2, this->e2r, tmp1, prec);
        // tmp3 = e2r + 1/e2r
        arb_add(tmp3, this->e2r, tmp1, prec);
        // lmd = tmp2 / tmp3
        arb_div(this->lmd, tmp2, tmp3, prec);
        // lmd = lmd * Deltam
        arb_mul(this->lmd, this->lmd, this->Deltam, prec);

        // Pre-calculate sin(theta) and cos(theta)
        arb_sin_cos(tmp_sin, tmp_cos, this->theta, prec);

        // 7. D = (lmd * sin(theta))^2 + omegap^2
        arb_mul(tmp1, this->lmd, tmp_sin, prec); // tmp1 = lmd * sin
        arb_mul(tmp1, tmp1, tmp1, prec);         // tmp1 = (lmd * sin)^2
        arb_mul(tmp2, this->omegap, this->omegap, prec); // tmp2 = omegap^2
        arb_add(this->D, tmp1, tmp2, prec);

        // 8. mu = lmd * cos(theta) + Deltam
        arb_mul(tmp1, this->lmd, tmp_cos, prec);
        arb_add(this->mu, tmp1, this->Deltam, prec);

        // 9. beta = Deltam - lmd * cos(theta)
        // Note: tmp1 still holds lmd * cos(theta)
        arb_sub(this->beta, this->Deltam, tmp1, prec);

        // 10. Omega = sqrt(beta * mu)
        arb_mul(tmp1, this->beta, this->mu, prec);
        arb_sqrt(this->Omega, tmp1, prec);

        //11. alpha
        set_alpha(dt);

        //12. delta
        set_delta(dt);

        // Clear temporary variables
        arb_clear(tmp1); arb_clear(tmp2);
        arb_clear(tmp3); arb_clear(tmp_sin); arb_clear(tmp_cos);

        std::cout << "\n--- Derived Quantities ---" << std::endl;
        print_arb("theta", theta);
        print_arb("r", r);
        print_arb("e2r", e2r);
        print_arb("dt", dt);
        print_arb("Deltam", Deltam);
        print_arb("lmd", lmd);
        print_arb("D", D);
        print_arb("mu", mu);
        print_arb("beta", beta);
        print_arb("Omega", Omega);
        print_arb("alpha",alpha);
        print_arb("delta",delta);
        std::cout << "--------------------------" << std::endl;






    }//end constructor

    // Destructor to clear arb_t memory
    ~table()
    {
        arb_clear(pi);
        arb_clear(half);
        arb_clear(quarter);
        arb_clear(two);
        arb_clear(alpha);
        arb_clear(delta);



        arb_clear(g0);
        arb_clear(omegam);
        arb_clear(omegap);
        arb_clear(omegac);
        arb_clear(er);
        arb_clear(thetaCoef);
        arb_clear(theta);
        arb_clear(lmd);
        arb_clear(Deltam);
        arb_clear(r);
        arb_clear(e2r);
        arb_clear(tTot);
        arb_clear(dt);
        arb_clear(D);
        arb_clear(mu);
        arb_clear(beta);
        arb_clear(Omega);

    }

public:

    void generate_table();

    ///
    /// @brief Parallel implementation of Z_tilde summation using C++11 threads
    /// @param result The arb_t variable to store the final summation
    /// @param j
    /// @param k
    /// @param n1
    /// @param n2
    void Z_tilde_parallel(arb_t result, int j, int k, int n1, int n2);
    ///
    /// @param result sequential summation
    /// @param j
    /// @param k
    /// @param n1
    /// @param n2
    void Z_tilde_sequential(arb_t result,int j,int k,int n1,int n2);

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
    void summation_one_term(arb_t result,int j,int k,int n1,int n2,int R, int m1, int m2, int m3, int m4,int t);

    ///
    /// @brief Set alpha = e^(lambda*sin(theta)*tau)
    /// @param tau time parameter
    void set_alpha(const arb_t tau);

    ///
    /// @brief Set delta using current alpha value
    /// @param tau time parameter
    void set_delta(const arb_t tau);

    ///
    /// @brief Set both alpha and delta for a given tau
    /// @param tau time parameter
    void set_alpha_delta(const arb_t tau);
    ///
    /// @param result pcfu
    /// @param a
    /// @param z
    void pcfu(arb_t result, const arb_t a, const arb_t z);

    std::string to_string_val( arb_t x, slong digits);

    std::string to_string_err( arb_t x, slong digits);

    // Helper function to print arb_t
    static void print_arb(const char* name, arb_t& val) {
        char* s = arb_get_str(val, 30, 0); // 15 digits for display
        std::cout << name << "=" << s<<"\n";

        flint_free(s);
    }


public:
    const slong prec = 500;
    const slong out_digits=30;
    int j1H;
    int j2H;
    arb_t pi;
    arb_t g0;
    arb_t omegam;
    arb_t omegap;
    arb_t omegac;
    arb_t er;
    arb_t thetaCoef;
    int groupNum;
    int rowNum;
    arb_t theta;
    arb_t lmd;
    arb_t Deltam;
    arb_t r;
    arb_t e2r;
    int N1;
    int N2;
    arb_t tTot;
    arb_t dt;
    int Q;
    arb_t D;
    arb_t mu;
    arb_t beta;
    arb_t Omega;

    arb_t quarter,half,two;

    arb_t alpha,delta;
    int num_threads ;
    std::string out_dir;
};





#endif //INTEGRAL_TABLE_HPP