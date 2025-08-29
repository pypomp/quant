# Makefile for measles jobs
# Usage: make gpu RUN_LEVEL=2 or make cpu RUN_LEVEL=2

# Declare phony targets (targets that don't represent files)
.PHONY: gpu cpu clean

# Default run level if not specified
RUN_LEVEL ?= 2

# Set wall time based on run level
# Level 1: 30 seconds, Level 2: 4 minutes, Level 3: 4 hours, Level 4: 12 hours
ifeq ($(RUN_LEVEL),1)
WALL_TIME := 00:01:00
else ifeq ($(RUN_LEVEL),2)
WALL_TIME := 00:30:00
else ifeq ($(RUN_LEVEL),3)
WALL_TIME := 04:00:00
else ifeq ($(RUN_LEVEL),4)
WALL_TIME := 12:00:00
else
$(warning Unknown RUN_LEVEL '$(RUN_LEVEL)', defaulting to 12:00:00)
WALL_TIME := 12:00:00
endif

# GPU job
gpu:
	cd gpu && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=04:00:00 job_gpu.sbat

# CPU job  
cpu:
	cd cpu && RUN_LEVEL=$(RUN_LEVEL) USE_CPU=true sbatch --time=$(WALL_TIME) job_cpu.sbat

report:
	cd .. && quarto render measles/report.qmd
