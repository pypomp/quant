# Makefile for sampler speed test

# Declare phony targets (targets that don't represent files)
.PHONY: gpu cpu clean

# Make logs directory
cpu_logs:
	mkdir -p cpu_logs

gpu_logs:
	mkdir -p gpu_logs

# GPU job
gpu: gpu_logs
	USE_CPU=false sbatch --time=00:02:00 job_gpu.sbat

# CPU job
cpu: cpu_logs
	USE_CPU=true sbatch --time=00:05:00 job_cpu.sbat

both: gpu cpu

clean:
	rm -rf cpu_logs gpu_logs