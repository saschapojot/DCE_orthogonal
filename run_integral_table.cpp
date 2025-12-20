#include "./compute_integral_table/integral_table.hpp"



int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }
    auto table_obj=table(std::string(argv[1]));

    int j=1;
   int k=5;
    int n1=3;
    int n2=1;
    std::cout<<"j="<<j<<", k="<<k<<", n1="<<n1<<", n2="<<n2<<std::endl;
    arb_t result;
    arb_init(result);
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    table_obj.Z_tilde_sequential(result,j,k,n1,n2);

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the result
    std::cout << "Computation time: " << duration.count() << " ms" << std::endl;
    table_obj.print_arb("result:",result);
    // Optional: Clear arb variable if required by your library conventions
    arb_clear(result);







}