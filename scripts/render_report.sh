#!/bin/bash
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=quarto_render
#SBATCH --output=scratch/quarto_render_%j.log

# Load R module
module load R/4.4.0

# Path to the .qmd file
REPORT=$1

# if [ -z "$REPORT" ]; then
#     echo "No path provided. Defaulting to tests/spx/report.qmd"
#     REPORT="tests/spx/report.qmd"
# fi

# Basic check if it exists
if [ ! -f "$REPORT" ]; then
    echo "Error: File $REPORT not found."
    exit 1
fi

echo "Rendering $REPORT on $(hostname)"
echo "Started at: $(date)"

quarto render "$REPORT"

echo "Finished render at: $(date)"
