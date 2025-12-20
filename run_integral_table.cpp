#include "./compute_integral_table/integral_table.hpp"



int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <number>" << std::endl;
        std::cerr << "Error: wrong arguments" << std::endl;
        std::exit(2);
    }
    auto table_obj=table(std::string(argv[1]));

    int proc_num = std::stoi(argv[2]);
    table_obj.num_threads=proc_num;
    int j=5;
    int k=2;
    int n1=5;
    int n2=2;
    std::cout<<"j="<<j<<", k="<<k<<", n1="<<n1<<", n2="<<n2<<std::endl;
    arb_t result;

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    table_obj.Z_tilde_parallel(result,j,k,n1,n2);


    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> elapsed = end - start;
    table_obj.print_arb("result",result);

    // Print execution time
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;
    arb_clear(result);







}