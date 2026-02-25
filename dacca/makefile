# Makefile for Dacca jobs
# Usage: make gpu RUN_LEVEL=2 or make cpu RUN_LEVEL=2

# Declare phony targets (targets that don't represent files)
.PHONY: gpu cpu train clean

# Default run level if not specified
RUN_LEVEL ?= 1

# Set wall time based on run level
ifeq ($(RUN_LEVEL),1)
WALL_TIME := 00:01:00
else ifeq ($(RUN_LEVEL),2)
WALL_TIME := 00:30:00
else ifeq ($(RUN_LEVEL),3)
WALL_TIME := 00:30:00
else ifeq ($(RUN_LEVEL),4)
WALL_TIME := 00:30:00
else
$(warning Unknown RUN_LEVEL '$(RUN_LEVEL)', defaulting to 12:00:00)
WALL_TIME := 12:00:00
endif

# Fits the Dacca model on the GPU using IF2 only.
# Results are saved under performance/gpu_results
performance_gpu:
	mkdir -p performance/gpu_results/logs
	cd performance && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=$(WALL_TIME) job_gpu.sbat

# Fits the Dacca model on the CPU using IF2 only
# Results are saved under performance/cpu_results
performance_cpu:
	mkdir -p performance/cpu_results/logs
	cd performance && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=true sbatch --time=$(WALL_TIME) job_cpu.sbat

# Fits the Dacca model on the GPU using IF2 + train
# Results are saved under performance/train_results
performance_train:
	mkdir -p performance/train_results/logs
	cd performance && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=$(WALL_TIME) job_train.sbat

# Run all performance tests
performance: performance_gpu performance_cpu performance_train

# Estimates the logLik of the Dacca model using pypomp. 
# Results are saved under pfilter_check.
eval:
	cd pfilter_check && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=00:20:00 eval_py.sbat

performance_report:
	cd .. && quarto render dacca/performance/report.qmd
