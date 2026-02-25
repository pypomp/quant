import argparse
import os
import re
import sys
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed. Please run `pip install PyYAML` or activate your .venv.")
    sys.exit(1)

CONFIG_START = "--- SLURM CONFIG ---"
CONFIG_END = "--- END SLURM CONFIG ---"

def parse_test_metadata(filepath):
    """Extracts and parses the YAML configuration from comments in the test script."""
    yaml_lines = []
    in_config = False
    
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            # Ignore completely empty lines
            if not stripped:
                continue
                
            # Both R and Python use '#' for comments
            if not stripped.startswith("#"):
                # If we hit actual code, stop looking
                break
            
            comment_content = stripped[1:].strip()
            
            if comment_content == CONFIG_START:
                in_config = True
                continue
            elif comment_content == CONFIG_END:
                in_config = False
                break
            elif in_config:
                # Keep the indentation relative to the comment start
                # We do this by replacing the first '#' and an optional space
                raw_line = line.lstrip()
                if raw_line.startswith("# "):
                    yaml_core = raw_line[2:]
                elif raw_line.startswith("#"):
                    yaml_core = raw_line[1:]
                else:
                    yaml_core = raw_line
                yaml_lines.append(yaml_core)

    if not yaml_lines:
        return None

    try:
        config = yaml.safe_load("".join(yaml_lines))
        return config if config else {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in {filepath}:\n{e}")
        sys.exit(1)


def generate_sbatch_script(test_filepath, config, run_level):
    """Generates the content for a dynamic sbatch script."""
    lines = ["#!/bin/bash"]
    
    # 1. Base SBATCH arguments
    name = config.get("name", os.path.basename(test_filepath))
    lines.append(f"#SBATCH --job-name={name}")
    
    sbatch_args = config.get("sbatch_args", {})
    
    # Merge run-level specific arguments
    if run_level and "run_levels" in config:
        rl_config = config["run_levels"].get(int(run_level)) or config["run_levels"].get(str(run_level))
        if rl_config and "sbatch_args" in rl_config:
            sbatch_args.update(rl_config["sbatch_args"])

    for key, value in sbatch_args.items():
        # Handle flags without values
        if value is True:
            lines.append(f"#SBATCH --{key}")
        elif value is not False:
            if isinstance(value, str) and " " in value:
                lines.append(f'#SBATCH --{key}="{value}"')
            else:
                lines.append(f"#SBATCH --{key}={value}")

    lines.append("")
    lines.append(f'echo "Running on $SLURM_JOB_NODELIST"')
    lines.append(f'echo "Running in $(pwd)"')
    lines.append("")
    
    # 2. Set environment variables (like RUN_LEVEL)
    if run_level:
        lines.append(f"export RUN_LEVEL={run_level}")
        
    for k, v in config.get("env", {}).items():
        lines.append(f"export {k}={v}")

    # 3. Setup block
    setup_commands = config.get("setup", "")
    if setup_commands:
        lines.append("\n### Setup =====")
        lines.append(setup_commands.strip())
        
    # Default behavior: run python/R script depending on extension
    test_filepath_abs = os.path.abspath(test_filepath)
    test_dir_abs = os.path.dirname(test_filepath_abs)
    
    # Navigate to the test directory so it runs in context natively
    lines.append("\n### Main ======")
    lines.append(f"cd {test_dir_abs}")
    
    # Base command logic
    command = config.get("command")
    if not command:
        ext = os.path.splitext(test_filepath)[1].lower()
        if ext == ".py":
            # Assume local virtualenv at the project root
            project_root = os.path.abspath(os.path.dirname(__file__))
            venv_bin = os.path.join(project_root, ".venv", "bin", "activate")
            lines.append(f"source {venv_bin}")
            command = f"python -u {os.path.basename(test_filepath)}"
        elif ext in [".r", ".R"]:
            project_root = os.path.abspath(os.path.dirname(__file__))
            activate_r = os.path.join(project_root, ".renv", "activate.R")
            lines.append(f'echo "source(\'{activate_r}\')" > .Rprofile')
            
            # Use R CMD BATCH to generate a .Rout file by default for R scripts
            script_name = os.path.basename(test_filepath)
            command = f"R CMD BATCH --no-restore --no-save {script_name} {script_name}out"
        else:
            print(f"Error: Unknown file type {ext} for {test_filepath}. Please provide a 'command' securely in YAML.")
            sys.exit(1)
            
    lines.append(command)
    
    return "\n".join(lines)


def run_test_config(filepath, config, run_level, dry_run=False):
    script_content = generate_sbatch_script(filepath, config, run_level)
    
    test_dir = os.path.dirname(os.path.abspath(filepath))
    
    if "sbatch_args" in config and "output" in config["sbatch_args"]:
        output_path = config["sbatch_args"]["output"]
        
        # Ensure log directories exist relative to the test
        # Handle both %j outputs and static ones
        if "%" in output_path:
            # If the filename has slurm substitution elements like %j, %x, etc.
            output_dir = os.path.dirname(output_path)
            if output_dir:
                 os.makedirs(os.path.join(test_dir, output_dir), exist_ok=True)
        else:
             output_dir = os.path.dirname(output_path)
             if output_dir:
                 os.makedirs(os.path.join(test_dir, output_dir), exist_ok=True)
            
    print(f"\n--- Submitting: {config.get('name', 'Job')} ({filepath}) ---")
    if dry_run:
        print("DRY RUN: Generated script:")
        print(script_content)
        print("-" * 30)
    else:
        temp_sbat = os.path.join(test_dir, f".temp_run_{config.get('name', 'job').replace(' ', '_')}.sbat")
        with open(temp_sbat, "w") as f:
            f.write(script_content)
            
        try:
            subprocess.run(["sbatch", os.path.basename(temp_sbat)], cwd=test_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit {filepath}: {e}")
        except FileNotFoundError:
            print("Error: 'sbatch' command not found. Are you running this on the SLURM cluster?")
        finally:
            if os.path.exists(temp_sbat):
                os.remove(temp_sbat)


def run_test(filepath, run_level, target_job=None, dry_run=False):
    config = parse_test_metadata(filepath)
    if not config or not isinstance(config, dict):
        print(f"Ignored: no valid SLURM CONFIG found in {filepath}")
        return

    # Support multiple jobs if specified under 'jobs'
    if "jobs" in config and isinstance(config["jobs"], dict):
        for job_name, job_config in config["jobs"].items():
            if not isinstance(job_config, dict):
                continue
            
            # If the user specified a target job, skip the others
            if target_job and target_job != job_name:
                continue

            # Merge base config into job config
            merged = config.copy()
            merged.pop("jobs")
            merged.update(job_config)
            merged["name"] = job_name
            
            # Merge sbatch args specifically
            if "sbatch_args" in config and "sbatch_args" in job_config:
                if isinstance(config["sbatch_args"], dict) and isinstance(job_config["sbatch_args"], dict):
                    merged["sbatch_args"] = config["sbatch_args"].copy()
                    merged["sbatch_args"].update(job_config["sbatch_args"])
                
            run_test_config(filepath, merged, run_level, dry_run)
    else:
        # For single job configs, just check if the name matches the target (if provided)
        job_name = config.get("name", os.path.basename(filepath))
        if target_job and target_job != job_name:
             return
             
        run_test_config(filepath, config, run_level, dry_run)


def find_tests(target_path):
    target_path = Path(target_path)
    test_files = []
    
    if target_path.is_file():
        test_files.append(str(target_path))
    elif target_path.is_dir():
        for ext in ["*.py", "*.R", "*.r"]:
            for file in target_path.rglob(ext):
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(2000)
                    if CONFIG_START in content:
                        test_files.append(str(file))
    return test_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized SLURM test runner")
    parser.add_argument("action", choices=["run", "list"], nargs="?", default="run", help="Action to perform: 'run' submits tests, 'list' shows discovered tests.")
    parser.add_argument("target", nargs="?", default=".", help="File or directory containing tests (e.g. 'spx/performance/test.py' or 'spx/')")
    parser.add_argument("--run-level", default=os.environ.get("RUN_LEVEL"), help="The RUN_LEVEL to use (will set the env var and configure dynamic YAML configs)")
    parser.add_argument("--job", default=None, help="Only run a specific job by name if the file defines multiple in 'jobs:' (e.g. 'cpu' or 'gpu')")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated sbatch script without submitting")
    
    args = parser.parse_args()
    
    # Simple hack to assume activation of venv to grab yaml if we couldn't before
    if "yaml" not in sys.modules:
        pass # Already exited at top of script if missing
        
    tests = find_tests(args.target)
    
    if not tests:
        print(f"No tests found in '{args.target}' with a '{CONFIG_START}' block.")
        sys.exit(0)
        
    if args.action == "list":
        print(f"\nDiscovered {len(tests)} test(s) in '{args.target}':")
        for test in tests:
            config = parse_test_metadata(test)
            if not config:
                continue
                
            print(f"\n- {test}")
            if "jobs" in config and isinstance(config["jobs"], dict):
                job_names = list(config["jobs"].keys())
                print(f"  Includes jobs: {', '.join(job_names)}")
            else:
                name = config.get("name", os.path.basename(test))
                print(f"  Includes job: {name}")
                
        print("\nUse 'python run_tests.py run <path>' to execute them.\n")
    elif args.action == "run":
        for test in tests:
            run_test(test, args.run_level, args.job, args.dry_run)
