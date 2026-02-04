# Makefile for sampler speed test

# Declare phony targets (targets that don't represent files)
.PHONY: gpu cpu clean

# Set wall time
WALL_TIME := 00:02:00

# Make logs directory
logs:
	mkdir -p logs

# GPU job
gpu: logs
	USE_CPU=false sbatch --time=$(WALL_TIME) job_gpu.sbat
