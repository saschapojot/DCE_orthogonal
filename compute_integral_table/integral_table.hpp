//
// Created by adada on 12/5/2025.
//

#ifndef INTEGRAL_TABLE_HPP
#define INTEGRAL_TABLE_HPP
#include <boost/filesystem.hpp>
// #include <boost/math/constants/constants.hpp>
// #include <boost/math/special_functions/gamma.hpp>
//// #include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/hypergeometric_2F0.hpp>
//#include <boost/multiprecision/cpp_complex.hpp>
//#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gsl/gsl_sf_hyperg.h> // GNU Scientific Library

#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
// const auto PI = boost::math::constants::pi<boost::multiprecision::cpp_dec_float_50>();
// using Complex50 = boost::multiprecision::cpp_complex_50;
// const auto I = Complex50(0, 1);
// Define the high-precision floating point type alias for easier usage
// using double = boost::multiprecision::cpp_dec_float_50;
// using boost::multiprecision::pow;
// using boost::multiprecision::sin;
// using boost::multiprecision::exp;
// using boost::multiprecision::log;
// using boost::multiprecision::cos;
const auto PI=M_PI;
using namespace std::complex_literals; // Brings in the i literal
namespace fs = boost::filesystem;
namespace bp = boost::python;
namespace np = boost::python::numpy;
class table
{
public:
    table(const std::string &cppInParamsFileName)
    {
        std::ifstream file(cppInParamsFileName);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0;
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
                iss>>g0;
                paramCounter++;
                continue;
            }//end reading g0

            //read omegam
            if(paramCounter == 3)
            {
                iss>>omegam;
                paramCounter++;
                continue;
            }//end reading omegam

            //read omegap
            if(paramCounter == 4)
            {
                iss>>omegap;
                paramCounter++;
                continue;
            }
            //end reading omegap
            //read omegac
            if(paramCounter == 5)
            {
                iss>>omegac;
                paramCounter++;
                continue;
            }//end reading omegac

            //read er
            if(paramCounter == 6)
            {
                iss>>er;
                if(er<=0)
                {
                    std::cerr << "er must be >0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }//end reading er

            //read thetaCoef
            if(paramCounter == 7)
            {
                iss>>thetaCoef;
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
                iss>>tTot;
                if (tTot<=0)
                {
                    std::cerr << "tTot must be >0" << std::endl;
                    std::exit(1);
                }
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
        std::cout<<"j1H="<<j1H<<", j2H="<<j2H<<", g0="<<g0
        <<", omegam="<<omegam<<", omegap="<<omegap<<", omegac="<<omegac
        <<", er="<<er<<", thetaCoef="<<thetaCoef<<", groupNum="
        <<groupNum<<", rowNum="<<rowNum<<", N1="<<N1<<", N2="<<N2<<", tTot="<<tTot<<", Q="<<Q<<std::endl;

 //compute derived quantities


        // 1. Calculate theta using the high-precision PI constant
        this->theta = this->thetaCoef * PI;
        // 2. Calculate r (squeezing parameter) using Boost Multiprecision log
        this->r = log(this->er);
        // 3. Calculate e^2r. Since er = e^r, then e^2r = (er)^2.
        this->e2r = this->er * this->er;
        // 4. Calculate time step dt
        this->dt = this->tTot / double(this->Q);
        //5. Deltam=omegam-omegap
        this->Deltam=omegam-omegap;
        //6. lambda
        this->lmd=(e2r-1.0/e2r)/(e2r+1.0/e2r)*Deltam;
        //7. D
        D=pow(lmd*sin(theta),2.0)+pow(omegap,2.0);
        //8. mu
        mu=lmd*cos(theta)+Deltam;
        //9. beta
        beta=Deltam-lmd*cos(theta);
        //10. Omega
        Omega=sqrt(beta*mu);

        // Print derived quantities
        std::cout << "\n--- Derived Quantities ---" << std::endl;
        std::cout << "theta   = " << theta << std::endl;
        std::cout << "r       = " << r << std::endl;
        std::cout << "e2r     = " << e2r << std::endl;
        std::cout << "dt      = " << dt << std::endl;
        std::cout << "Deltam  = " << Deltam << std::endl;
        std::cout << "lmd     = " << lmd << std::endl;
        std::cout << "D       = " << D << std::endl;
        std::cout << "mu      = " << mu << std::endl;
        std::cout << "beta    = " << beta << std::endl;
        std::cout << "Omega   = " << Omega << std::endl;
        std::cout << "--------------------------" << std::endl;


    }//end constructor
public:
    ///
    /// @param a
    /// @param z
    /// @return Parabolic Cylinder Function U(a,z)
    double pcf_U(const double& a, const double& z);

    ///
    /// @param nu
    /// @param x
    /// @return Function to calculate D_nu(x) using GSL's Hypergeometric U
    double pcf_D(const double& nu, const double&  x);
public:
    int j1H;
    int j2H;
     double g0;
    double omegam;
    double omegap;
    double omegac ;
    double er ;
    double thetaCoef ;
    int groupNum ;
    int rowNum ;
    double theta;
    double lmd;
    double Deltam;
    double r;
    double e2r;
    int N1;
    int N2;
    double tTot;
    double dt;
    int Q;
    double D;
    double mu;
    double beta;
    double Omega;
};





#endif //INTEGRAL_TABLE_HPP
