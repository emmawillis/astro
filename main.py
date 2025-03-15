
import submitit

def aaaa():
    print("hiiiiii") # TODO 

executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=10)
executor.update_parameters(
    slurm_gres='gpu:a40:1', 
    cpus_per_task=16,
    slurm_time=120,  # Increase time limit to 2 hours (in minutes)
    stderr_to_stdout=True,
    slurm_name="emma test"
)

job = executor.submit(aaaa)
print(job.job_id)

output = job.result()  # waits for completion and returns output
print("done. output: ", output)