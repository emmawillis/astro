
import submitit
from train import train

executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=10)
executor.update_parameters(
    slurm_gres='gpu:a40:1', 
    cpus_per_task=16,
    slurm_time=580,
    stderr_to_stdout=True,
    slurm_name="200-astro-emma"
)

job = executor.submit(train)
print(job.job_id)

output = job.result()  # waits for completion and returns output
print("done. output: ", output)