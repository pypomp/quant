# Makefile for SPX jobs
# Usage: make gpu RUN_LEVEL=2 or make cpu RUN_LEVEL=2

# Declare phony targets (targets that don't represent files)
.PHONY: gpu cpu clean

# Default run level if not specified
RUN_LEVEL ?= 1

# Set wall time based on run level
ifeq ($(RUN_LEVEL),1)
WALL_TIME := 00:00:30
else ifeq ($(RUN_LEVEL),2)
WALL_TIME := 00:04:00
else ifeq ($(RUN_LEVEL),3)
WALL_TIME := 02:00:00
else ifeq ($(RUN_LEVEL),4)
WALL_TIME := 06:00:00
else
$(warning Unknown RUN_LEVEL '$(RUN_LEVEL)', defaulting to 12:00:00)
WALL_TIME := 12:00:00
endif

# GPU job
performance_gpu:
	mkdir -p performance/gpu_results/logs
	cd performance && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=$(WALL_TIME) job_gpu.sbat

# CPU job  
performance_cpu:
	mkdir -p performance/cpu_results/logs
	cd performance && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=true sbatch --time=$(WALL_TIME) job_cpu.sbat

performance_R:
	mkdir -p performance/R_results/logs
	cd .. && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=true sbatch --time=03:20:00 spx/performance/job_R.sbat

# Eval pfilter with pypomp
py_pfilter_check:
	mkdir -p pfilter_check/py_results/logs
	cd pfilter_check && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=00:02:00 eval_py.sbat

# Eval pfilter with pomp (runs from root for renv)
R_pfilter_check:
	mkdir -p pfilter_check/R_results/logs
	cd .. && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=true sbatch --time=00:20:00 spx/pfilter_check/eval_R.sbat

# Run all pfilter checks
pfilter_check: py_pfilter_check R_pfilter_check

report:
	cd .. && quarto render spx/report.qmd
