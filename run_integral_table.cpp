#include "./compute_integral_table/integral_table.hpp"



int main(int argc, char *argv[])
{
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <filename> <number>" << std::endl;
        std::cerr << "Error: wrong arguments" << std::endl;
        std::exit(2);
    }
    auto table_obj=table(std::string(argv[1]));
    fs::path full_path(argv[1]);
    table_obj.out_dir=full_path.parent_path().string();
    std::cout<<"out_dir="<<table_obj.out_dir<<std::endl;
    int j_start=std::stoi(argv[2]);
    int j_end=std::stoi(argv[3]);
    int k_start=std::stoi(argv[4]);
    int k_end=std::stoi(argv[5]);
    int proc_num = std::stoi(argv[6]);

    table_obj.j_start=j_start;
    table_obj.j_end=j_end;
    table_obj.k_start=k_start;
    table_obj.k_end=k_end;
    table_obj.num_threads=proc_num;


    table_obj.generate_table();






    // int j=5;
    // int k=2;
    // int n1=5;
    // int n2=2;
    // std::cout<<"j="<<j<<", k="<<k<<", n1="<<n1<<", n2="<<n2<<std::endl;
    // arb_t result;
    //
    // // Start timer
    // auto start = std::chrono::high_resolution_clock::now();
    // table_obj.Z_tilde_parallel(result,j,k,n1,n2);
    //
    //
    // // Stop timer
    // auto end = std::chrono::high_resolution_clock::now();
    //
    // // Calculate duration
    // std::chrono::duration<double> elapsed = end - start;
    // table_obj.print_arb("result",result);
    //
    // // Print execution time
    // std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;
    // arb_clear(result);







}