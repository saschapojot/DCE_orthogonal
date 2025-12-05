//
// Created by adada on 12/5/2025.
//

#ifndef INTEGRAL_TABLE_HPP
#define INTEGRAL_TABLE_HPP
#include <boost/filesystem.hpp>
#include <boost/math/constants/constants.hpp>



#include <boost/math/special_functions/gamma.hpp>

#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/multiprecision/cpp_complex.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

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
const auto PI=boost::math::constants::pi;
using Complex50 = boost::multiprecision::cpp_complex_50;
const auto I = Complex50(0, 1);



namespace fs = boost::filesystem;
namespace bp = boost::python;
namespace np = boost::python::numpy;






#endif //INTEGRAL_TABLE_HPP
