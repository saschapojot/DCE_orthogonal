import os
import sys
import pandas as pd
from pathlib import Path
import shutil
#python readCSV.py groupNum rowNum
#this script reads csv and creates directory
if len(sys.argv)!=3:
    print("wrong number of arguments")

groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])

inParamFileName="./inParams/inParams"+str(groupNum)+".csv"
# print("file name is "+inParamFileName)
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]

N1=int(oneRow.loc["N1"])
N2=int(oneRow.loc["N2"])

print(f"N1={N1}, N2={N2}")

def generate_job_params(N1, N2, num_jobs_j, num_jobs_k):
    """
    Generate job parameters for dividing the computation space.
    :param N1: Total range for j (0 to N1-1)
    :param N2: Total range for k (0 to N2-1)
    :param num_jobs_j: Number of chunks to divide j range into
    :param num_jobs_k: Number of chunks to divide k range into
    :return: List of tuples: [(job_id, j_start, j_end, k_start, k_end), ...]
    """

    # Calculate chunk sizes
    j_chunk_size = N1 // num_jobs_j
    k_chunk_size = N2 // num_jobs_k
    job_params = []
    job_id = 0

    # Generate all combinations
    for j_chunk in range(num_jobs_j):
        j_start = j_chunk * j_chunk_size
        # Handle remainder for last chunk
        if j_chunk == num_jobs_j - 1:
            j_end = N1
        else:
            j_end = (j_chunk + 1) * j_chunk_size
        for k_chunk in range(num_jobs_k):
            k_start = k_chunk * k_chunk_size
            # Handle remainder for last chunk
            if k_chunk == num_jobs_k - 1:
                k_end = N2
            else:
                k_end = (k_chunk + 1) * k_chunk_size
            job_params.append((job_id, j_start, j_end, k_start, k_end))
            job_id += 1

    return job_params

def write_slurm_script(job_id, j_start, j_end, k_start, k_end,
                       groupNum, rowNum,out_dir,
                       num_threads=24,
                       time_limit="0-60:00",
                       mem="100GB",
                       partition="lzicnormal",
                       base_dir="/public/home/hkust_jwliu_1/liuxi/Documents/cppCode/DCE_orthogonal"):
    """
   Write a SLURM bash script for a single job.

   Args:
       job_id: Job identifier
       j_start, j_end: j range for this job
       k_start, k_end: k range for this job
       groupNum: Group number
       rowNum: Row number
       num_threads: Number of CPU threads
       time_limit: SLURM time limit
       mem: Memory allocation
       partition: SLURM partition name
       base_dir: Base directory for the job

   Returns:
       filename: Name of the created bash script
   """

    input_file = f"./outData/group{groupNum}/row{rowNum}/cppIn.txt"

    script_content = [
        "#!/bin/bash",
        "#SBATCH -n 1",
        "#SBATCH -N 1",
        f"#SBATCH -t {time_limit}",
        f"#SBATCH --cpus-per-task={num_threads}",
        f"#SBATCH -p {partition}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -o out_group{groupNum}_row{rowNum}_job{job_id}.out",
        f"#SBATCH -e out_group{groupNum}_row{rowNum}_job{job_id}.err",
        f"#SBATCH -J g{groupNum}r{rowNum}j{job_id}",
        "",
        f"cd {base_dir}",
        "",
        f'echo "Job {job_id}: j=[{j_start},{j_end}), k=[{k_start},{k_end})"',
        'echo "Started at: $(date)"',
        "",
        f"python3 -u readCSV_createDir.py {groupNum} {rowNum}",
        "",
        f"# Usage: run_integral_table <filename> <j_start> <j_end> <k_start> <k_end> <num_threads>",
        f"./run_integral_table {input_file} {j_start} {j_end} {k_start} {k_end} {num_threads}",
        "",
        'echo "Finished at: $(date)"'
    ]

    Path(out_dir).mkdir(exist_ok=True,parents=True)
    script_filename=out_dir+f"/job_group{groupNum}_row{rowNum}_id{job_id}.sh"
    with open(script_filename, 'w') as f:
        f.write('\n'.join(script_content))
    print(f"Created: {script_filename}")



out_dir="./slurm_files_gansu/"
# Remove output directory if it exists
if os.path.exists(out_dir):
    print(f"Removing existing directory: {out_dir}")
    shutil.rmtree(out_dir)


# Configure job division
num_jobs_j = 10  # Divide j into chunks
num_jobs_k = 10  # Divide k into chunks
# Generate job parameters
job_params = generate_job_params(N1, N2, num_jobs_j, num_jobs_k)
print(f"Generating {len(job_params)} jobs...")
print(f"Each job covers approximately {N1//num_jobs_j} j-values and {N2//num_jobs_k} k-values\n")
# Write individual SLURM scripts
for job_id, j_start, j_end, k_start, k_end in job_params:
    write_slurm_script(job_id, j_start, j_end, k_start, k_end,
                       groupNum, rowNum, out_dir,
                       num_threads=24,
                       time_limit="0-60:00",
                       mem="100GB",
                       partition="lzicnormal")