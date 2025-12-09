#include "./compute_integral_table/integral_table.hpp"



int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }
    auto table_obj=table(std::string(argv[1]));



    double a = 2.0;
    double x = 3.0;

    double rst=table_obj.pcf_U(a,x);
    std::cout<<rst<<std::endl;



}