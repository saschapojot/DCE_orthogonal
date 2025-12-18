#include "./compute_integral_table/integral_table.hpp"



int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }
    auto table_obj=table(std::string(argv[1]));

    arb_t a_param, z_param,rst;
    arb_init(a_param);
    arb_init(z_param);
    arb_init(rst);

    arb_set_d(a_param,2);
    arb_set_d(z_param,1);
    table_obj.pcfu(rst,a_param,z_param);

    table_obj.print_arb("rst",rst);

    arb_clear(a_param);
    arb_clear(z_param);
    arb_clear(rst);









}