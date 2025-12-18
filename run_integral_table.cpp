#include "./compute_integral_table/integral_table.hpp"



int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }
    auto table_obj=table(std::string(argv[1]));



    arb_t x;

    // 2. Initialize memory (Essential!)
    arb_init(x);

    // 3. Set a value (e.g., set x to Pi with 64 bits of precision)
    slong precision = 64;
    arb_const_pi(x, precision);
    // 4. Print the value
    // arb_get_str(variable, digits_to_print, flags)
    // It returns a C-string (char*) that must be freed later.
    char* str = arb_get_str(x, 15, 0);
    std::cout << "Value of x: " << str << std::endl;

    // 5. Free the string memory
    flint_free(str);

    // 6. Clear the arb variable memory (Essential!)
    arb_clear(x);





}