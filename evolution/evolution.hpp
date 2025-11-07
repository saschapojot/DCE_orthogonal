//
// Created by adada on 11/6/2025.
//

#ifndef EVOLUTION_HPP
#define EVOLUTION_HPP
#include <armadillo>
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/hermite.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>

#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
const auto PI=M_PI;

namespace fs = boost::filesystem;
using namespace std::complex_literals; // Brings in the i literal
namespace bp = boost::python;
namespace np = boost::python::numpy;
//This subroutine computes evolution using operator splitting + spectral method
//one step is exact solution of quasi-linear pde

class evolution
{
public:
    evolution(const std::string &cppInParamsFileName)
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
            }
            //end reading j1H

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

            }
            //end reading j2H

            //read g0
            if (paramCounter == 2)
            {
                iss>>g0;
                paramCounter++;
                continue;
            }
            //end reading g0
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
            }
            //end reading omegac
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
            }
            //end reading er
            //read thetaCoef
            if(paramCounter == 7)
            {
                iss>>thetaCoef;
                paramCounter++;
                continue;
            }
            //end reading thetaCoef
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
        this->r=std::log(er);
        this->theta=thetaCoef*PI;
        this->Deltam=omegam-omegap;
        std::cout<<"Deltam="<<Deltam<<std::endl;
        this->e2r=std::pow(er,2.0);

        this->lmd=(e2r-1/e2r)/(e2r+1/e2r)*Deltam;
        std::cout<<"lambda="<<lmd<<std::endl;

        D=std::pow(lmd*std::sin(theta),2.0)+std::pow(omegap,2.0);
        mu=lmd*std::cos(theta)+Deltam;
        beta=Deltam-lmd*std::cos(theta);
        std::cout<<"D="<<D<<std::endl;
        std::cout<<"mu="<<mu<<std::endl;
        std::cout<<"beta="<<beta<<std::endl;

        this->dt=tTot/static_cast<double>(Q);
        std::cout<<"dt="<<dt<<std::endl;
        Omega=std::sqrt(beta*mu);
        std::cout<<"Omega="<<Omega<<std::endl;
        //initialize B matrix
        B=arma::zeros<arma::cx_dmat>(N1,N2);
        // std::cout << "rows: " << B.n_rows << std::endl;
        // std::cout << "cols: " << B.n_cols << std::endl;
        // std::cout << "total elements: " << B.n_elem << std::endl;
        B(j1H,j2H)=std::complex<double>(1,0);



    }//end constructor
public:
    ///
    /// @param n order of Hermite function
    /// @param x variable
    /// @param omega frequency
    /// @return Hermite function (normalized)
    double harmonic_oscillator_wavefunction(int n, double x, double omega);
    ///
    /// @param n order
    /// @param x variable
    /// @return hermite polynomial value h_n(x)
    double hermite_polynomial(int n, double x) ;
    ///
    /// @param x1 variable for photon harmonic oscillator
    /// @param tau time step
    /// @return auxiliary function Delta for computing tensor Z
    double Delta(double x1, double tau);
    ///
    /// @param x1 variable for photon harmonic oscillator
    /// @return auxiliary function
    double rho(double x1);
    ///
    /// @param tau time step
    /// @return auxiliary function alpha for computing tensor Z
    double alpha(double tau);

    ///for testing
    void print_anything();
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
    //double dtEst;
    double tTot;
    double dt;
    int Q;



    double D;
    double mu;
    double beta;
    double Omega;
    arma::cx_dmat B;

};













#endif //EVOLUTION_HPP
