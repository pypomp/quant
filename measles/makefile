# Makefile for measles jobs
# Usage: make gpu RUN_LEVEL=2 or make cpu RUN_LEVEL=2

# Declare phony targets (targets that don't represent files)
.PHONY: gpu cpu clean

# Default run level if not specified
RUN_LEVEL ?= 1

# Set wall time based on run level
# Level 1: 30 seconds, Level 2: 4 minutes, Level 3: 4 hours, Level 4: 12 hours
ifeq ($(RUN_LEVEL),1)
WALL_TIME := 00:10:00
else ifeq ($(RUN_LEVEL),2)
WALL_TIME := 00:20:00
else ifeq ($(RUN_LEVEL),3)
WALL_TIME := 00:15:00
else ifeq ($(RUN_LEVEL),4)
WALL_TIME := 00:20:00
else
$(warning Unknown RUN_LEVEL '$(RUN_LEVEL)', defaulting to 12:00:00)
WALL_TIME := 12:00:00
endif

# GPU job
gpu:
	cd gpu && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=$(WALL_TIME) job_gpu.sbat

# CPU job  
cpu:
	cd cpu && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=true sbatch --time=12:00:00 job_cpu.sbat

# Eval pfilter with pypomp
eval:
	cd pfilter_check && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=00:20:00 eval_py.sbat

# Eval pfilter with pomp (runs from root for renv)
R_eval:
	cd .. && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=true sbatch --time=02:00:00 measles/R/measles.sbat

pypomp_eval:
	cd R && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=4:20:00 job_gpu.sbat

R_parameter_comparison:
	cd .. && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=true sbatch --time=15:00:00 measles/R/parameter_comparison/measles.sbat

py_parameter_comparison:
	cd R/parameter_comparison && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=02:30:00 job_gpu.sbat

report:
	cd .. && quarto render measles/report.qmd

R_report:
	cd .. && quarto render measles/R/report.qmd

R_parameter_comparison_report:
	cd .. && quarto render measles/R/parameter_comparison/report.qmd