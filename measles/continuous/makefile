# Makefile for continuous measles jobs
# Usage: make IF2 RUN_LEVEL=2 or make IFAD RUN_LEVEL=2

# Declare phony targets (targets that don't represent files)
.PHONY: IF2 IFAD clean

# Default run level if not specified
RUN_LEVEL ?= 1

# Set wall time based on run level
ifeq ($(RUN_LEVEL),1)
WALL_TIME := 00:03:00
else ifeq ($(RUN_LEVEL),2)
WALL_TIME := 00:20:00
else ifeq ($(RUN_LEVEL),3)
WALL_TIME := 00:30:00
else ifeq ($(RUN_LEVEL),4)
WALL_TIME := 01:00:00
else
$(warning Unknown RUN_LEVEL '$(RUN_LEVEL)', defaulting to 12:00:00)
WALL_TIME := 12:00:00
endif

IF2:
	mkdir -p IF2_results/logs
	RUN_LEVEL=$(RUN_LEVEL) sbatch --time=$(WALL_TIME) IF2_test.sbat

IFAD:
	mkdir -p IFAD_results/logs
	RUN_LEVEL=$(RUN_LEVEL) sbatch --time=$(WALL_TIME) IFAD_test.sbat

report:
	quarto render report.qmd