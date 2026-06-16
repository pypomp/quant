import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "Error: PyYAML is not installed. Please run `pip install PyYAML` or activate your .venv."
    )
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
                # If we hit actual code, keep looking for the config start
                # unless we were already in the config (which shouldn't happen)
                if in_config:
                    break
                continue

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


def load_global_config():
    """Loads a global test_config.yaml from the project root if it exists."""
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(project_root, "test_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse {config_path}:\n{e}")
    return {}


def parse_slurm_time(time_str):
    """Parses a SLURM time string (DD-HH:MM:SS, HH:MM:SS, MM:SS, or MM) into seconds."""
    days = 0
    has_days = False
    if "-" in time_str:
        parts = time_str.split("-", 1)
        days = int(parts[0])
        time_str = parts[1]
        has_days = True
    
    parts = list(map(int, time_str.split(":")))
    if has_days:
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, minutes = parts
            seconds = 0
        elif len(parts) == 1:
            hours = parts[0]
            minutes = 0
            seconds = 0
        else:
            raise ValueError(f"Invalid SLURM time format: {time_str}")
    else:
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        elif len(parts) == 1:
            hours = 0
            minutes = parts[0]
            seconds = 0
        else:
            raise ValueError(f"Invalid SLURM time format: {time_str}")
            
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_slurm_time(total_seconds):
    """Formats total seconds into a SLURM time string (DD-HH:MM:SS or HH:MM:SS)."""
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def generate_sbatch_script(test_filepath, config, run_level):
    """Generates the content for a dynamic sbatch script."""
    lines = ["#!/bin/bash"]

    # 1. Base SBATCH arguments
    name = config.get("name", os.path.basename(test_filepath))
    lines.append(f"#SBATCH --job-name={name}")

    sbatch_args = config.get("sbatch_args", {}).copy()

    # Merge run-level specific arguments
    if run_level and "run_levels" in config:
        rl_config = config["run_levels"].get(int(run_level)) or config[
            "run_levels"
        ].get(str(run_level))
        if rl_config and "sbatch_args" in rl_config:
            sbatch_args.update(rl_config["sbatch_args"])

    # Scale time based on N_UNITS if present
    env_config = config.get("env", {})
    n_units_val = env_config.get("N_UNITS") or os.environ.get("N_UNITS")
    if n_units_val and "time" in sbatch_args:
        try:
            n_units = int(n_units_val)
            original_time = sbatch_args["time"]
            scale_factor = max(1.0, n_units / 20.0)
            total_seconds = parse_slurm_time(original_time)
            scaled_seconds = int(total_seconds * scale_factor)
            scaled_time = format_slurm_time(scaled_seconds)
            sbatch_args["time"] = scaled_time
            print(f"Scaled time limit for {n_units} units: {original_time} -> {scaled_time}")
        except Exception as e:
            print(f"Warning: Failed to scale time '{sbatch_args.get('time')}' for N_UNITS={n_units_val}: {e}")

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
    lines.append('echo "Running on $SLURM_JOB_NODELIST"')
    lines.append('echo "Running in $(pwd)"')
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

    # Set up environment based on file extension
    ext = os.path.splitext(test_filepath)[1].lower()
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if ext == ".py":
        # Assume local virtualenv at the project root (one level up from scripts/)
        venv_bin = os.path.join(project_root, ".venv", "bin", "activate")
        lines.append(f"source {venv_bin}")
    elif ext in [".r", ".R"]:
        activate_r = os.path.join(project_root, ".renv", "activate.R")
        # Use RENV_PROJECT, RENV_PATHS_RENV and R_PROFILE_USER to point to the root renv without creating a local .Rprofile
        lines.append(f'export RENV_PROJECT="{project_root}"')
        lines.append(
            f'export RENV_PATHS_RENV="{os.path.join(project_root, ".renv")}"'
        )
        lines.append(f"export R_PROFILE_USER='{activate_r}'")

    # Base command logic
    command = config.get("command")
    if not command:
        if ext == ".py":
            command = f"python -u {os.path.basename(test_filepath)}"
        elif ext in [".r", ".R"]:
            # Use R CMD BATCH to generate a .Rout file by default for R scripts
            script_name = os.path.basename(test_filepath)
            command = (
                f"R CMD BATCH --no-restore --no-save {script_name} {script_name}out"
            )
        else:
            print(
                f"Error: Unknown file type {ext} for {test_filepath}. Please provide a 'command' securely in YAML."
            )
            sys.exit(1)

    lines.append(command)

    return "\n".join(lines)


def run_test_config(filepath, config, run_level, global_config=None, dry_run=False):
    # Merge global sbatch args into the specific job's config
    if global_config and "sbatch_args" in global_config:
        if "sbatch_args" not in config:
            config["sbatch_args"] = {}
        # We only want to add things from global_config that aren't ALREADY in config,
        # or we can overwrite. Standard behavior: Local overrides global.
        for k, v in global_config["sbatch_args"].items():
            if k not in config["sbatch_args"]:
                config["sbatch_args"][k] = v

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
        temp_sbat = os.path.join(
            test_dir, f".temp_run_{config.get('name', 'job').replace(' ', '_')}.sbat"
        )
        with open(temp_sbat, "w") as f:
            f.write(script_content)

        try:
            subprocess.run(
                ["sbatch", os.path.basename(temp_sbat)], cwd=test_dir, check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit {filepath}: {e}")
        except FileNotFoundError:
            print(
                "Error: 'sbatch' command not found. Are you running this on the SLURM cluster?"
            )
        finally:
            if os.path.exists(temp_sbat):
                os.remove(temp_sbat)


def run_test(filepath, run_level, target_job=None, global_config=None, dry_run=False):
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
                if isinstance(config["sbatch_args"], dict) and isinstance(
                    job_config["sbatch_args"], dict
                ):
                    merged["sbatch_args"] = config["sbatch_args"].copy()
                    merged["sbatch_args"].update(job_config["sbatch_args"])

            run_test_config(filepath, merged, run_level, global_config, dry_run)
    else:
        # For single job configs, just check if the name matches the target (if provided)
        job_name = config.get("name", os.path.basename(filepath))
        if target_job and target_job != job_name:
            return

        run_test_config(filepath, config, run_level, global_config, dry_run)


def find_tests(target_path):
    target_path = Path(target_path)
    test_files = []

    if target_path.is_file():
        test_files.append(str(target_path))
    elif target_path.is_dir():
        for ext in ["*.py", "*.R", "*.r", "*.qmd"]:
            for file in target_path.rglob(ext):
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(2000)
                    if CONFIG_START in content:
                        test_files.append(str(file))
    return test_files


def extract_fallback_description(filepath):
    """Fallback parser to extract docstring/Roxygen comments for descriptions."""
    ext = os.path.splitext(filepath)[1].lower()
    desc_lines = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if ext == ".py":
            in_docstring = False
            docstring_char = ""
            for line in lines:
                stripped = line.strip()
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_docstring = True
                        docstring_char = stripped[:3]
                        content = stripped[3:]
                        if docstring_char and content.endswith(docstring_char) and len(content) >= 3:
                            desc_lines.append(content[:-3].strip())
                            break
                        if content:
                            desc_lines.append(content.strip())
                else:
                    if docstring_char and stripped.endswith(docstring_char):
                        content = stripped[:-3]
                        if content:
                            desc_lines.append(content.strip())
                        break
                    desc_lines.append(line.strip())
        elif ext in [".r", ".R"]:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#'"):
                    desc_lines.append(stripped[2:].strip())
                elif stripped and not stripped.startswith("#"):
                    break
    except Exception:
        pass

    desc_lines = [line_str for line_str in desc_lines if line_str]
    if desc_lines:
        return desc_lines[0]
    return "No description provided"


def filter_tests(tests, importance_filter=None, tag_filter=None):
    """Filters tests based on importance level and tag."""
    filtered = []
    for test in tests:
        config = parse_test_metadata(test)
        if not config or not isinstance(config, dict):
            continue

        importance = config.get("importance", "").strip().lower()
        if not importance:
            importance = "low"

        tags = config.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip().lower() for t in tags.split(",") if t.strip()]
        elif isinstance(tags, list):
            tags = [str(t).strip().lower() for t in tags]
        else:
            tags = []

        if importance_filter:
            target_imp = importance_filter.strip().lower()
            imp_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            test_level = imp_levels.get(importance, 1)
            filter_level = imp_levels.get(target_imp, 1)
            if test_level < filter_level:
                continue

        if tag_filter:
            target_tag = tag_filter.strip().lower()
            if target_tag not in tags:
                continue

        filtered.append((test, config, importance, tags))
    
    # Sort tests alphabetically by path to group them by directory naturally
    filtered.sort(key=lambda x: x[0])
    return filtered


def get_category_name(filepath):
    """Helper to extract top-level test folder name as an uppercase category."""
    path = Path(filepath)
    parts_list = list(path.parts)
    if not parts_list:
        return "GENERAL"
    if parts_list[0] == "tests" and len(parts_list) > 1:
        return parts_list[1].upper()
    return parts_list[0].upper()


def print_tests_table(filtered_tests):
    """Prints a beautiful, formatted table/list of the tests grouped by category."""
    print("=" * 100)
    print(f"  {'#':<3}  {'Importance':<10}  {'Test Path / Description & Details'}")
    print("=" * 100)
    
    current_category = None
    for idx, (filepath, config, importance, tags) in enumerate(filtered_tests, 1):
        category = get_category_name(filepath)
        
        # Check if we are transitioning to a new category parent folder
        if category != current_category:
            if current_category is not None:
                # Double divider line when transitioning to a new parent folder
                print("=" * 100)
            print(f"\n--- {category} {'-' * (95 - len(category))}")
            current_category = category
        elif idx > 1:
            # Simple divider within the same category
            print("-" * 100)

        desc = config.get("description")
        if not desc:
            desc = extract_fallback_description(filepath)
        else:
            desc = desc.strip()

        if "jobs" in config and isinstance(config["jobs"], dict):
            jobs_str = ", ".join(config["jobs"].keys())
        else:
            jobs_str = config.get("name", os.path.basename(filepath))

        tags_str = ", ".join(tags) if tags else "None"

        import textwrap

        color_start = ""
        color_end = ""
        if sys.stdout.isatty():
            if importance == "critical":
                color_start = "\033[91;1m"  # bright red bold
            elif importance == "high":
                color_start = "\033[31;1m"  # red bold
            elif importance == "medium":
                color_start = "\033[33;1m"  # yellow bold
            else:
                color_start = "\033[36m"    # cyan
            color_end = "\033[0m"

        imp_display = f"{color_start}{importance.upper():<10}{color_end}"

        print(f"  {idx:<3}  {imp_display}  {filepath}")
        
        # Wrap description to avoid going off screen
        wrapped_desc = textwrap.wrap(desc, width=75)
        if wrapped_desc:
            print(f"       {'Description:':<12} {wrapped_desc[0]}")
            for line in wrapped_desc[1:]:
                print(f"       {'':<12} {line}")
        else:
            print(f"       {'Description:':<12} ")
            
        print(f"       {'Jobs:':<12} {jobs_str}")
        print(f"       {'Tags:':<12} {tags_str}")
    print("=" * 100)


def run_interactive_mode(filtered_tests, global_config, dry_run):
    """Prompts the user to interactively select and run tests."""
    if not filtered_tests:
        print("No tests found matching the criteria.")
        return

    print("\nInteractive Test Runner")
    print("=======================")
    print_tests_table(filtered_tests)

    run_level = os.environ.get("RUN_LEVEL")
    if not run_level:
        try:
            level_input = input("\nEnter RUN_LEVEL (e.g. 1, 2, 3, 4) [default: 1]: ").strip()
            if level_input:
                run_level = level_input
            else:
                run_level = "1"
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return

    try:
        user_input = input("\nEnter test numbers/ranges to run (e.g., '1, 3', '1-3', 'all', or 'q' to quit): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if not user_input or user_input.lower() in ["q", "quit"]:
        print("Cancelled.")
        return

    selected_indices = set()
    num_tests = len(filtered_tests)

    if user_input.lower() == "all":
        selected_indices = set(range(1, num_tests + 1))
    else:
        parts = [p.strip() for p in user_input.split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                try:
                    start_str, end_str = part.split("-", 1)
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    if start <= end:
                        for idx in range(start, end + 1):
                            if 1 <= idx <= num_tests:
                                selected_indices.add(idx)
                    else:
                        print(f"Warning: Invalid range '{part}' ignored.")
                except ValueError:
                    print(f"Warning: Invalid range format '{part}' ignored.")
            else:
                try:
                    idx = int(part)
                    if 1 <= idx <= num_tests:
                        selected_indices.add(idx)
                    else:
                        print(f"Warning: Index {idx} out of range (1-{num_tests}) ignored.")
                except ValueError:
                    print(f"Warning: Invalid selection '{part}' ignored.")

    if not selected_indices:
        print("No valid tests selected. Exiting.")
        return

    print(f"\nRunning {len(selected_indices)} selected test(s) at RUN_LEVEL={run_level}:")
    for idx in sorted(selected_indices):
        filepath, config, importance, tags = filtered_tests[idx - 1]
        run_test(filepath, run_level, target_job=None, global_config=global_config, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized SLURM test runner")
    parser.add_argument(
        "action",
        choices=["run", "list"],
        nargs="?",
        default="run",
        help="Action to perform: 'run' submits tests, 'list' shows discovered tests.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="File or directory containing tests (e.g. 'spx/performance/test.py' or 'spx/')",
    )
    parser.add_argument(
        "--run-level",
        default=os.environ.get("RUN_LEVEL"),
        help="The RUN_LEVEL to use (will set the env var and configure dynamic YAML configs)",
    )
    parser.add_argument(
        "--job",
        default=None,
        help="Only run a specific job by name if the file defines multiple in 'jobs:' (e.g. 'cpu' or 'gpu')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated sbatch script without submitting",
    )
    parser.add_argument(
        "--importance",
        "-p",
        default=None,
        help="Filter tests by importance: low, medium, high, critical. Shows/runs tests at or above this level.",
    )
    parser.add_argument(
        "--tag",
        "-t",
        default=None,
        help="Filter tests by a specific tag.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode to select tests to run.",
    )

    args = parser.parse_args()

    # Simple hack to assume activation of venv to grab yaml if we couldn't before
    if "yaml" not in sys.modules:
        pass  # Already exited at top of script if missing

    tests = find_tests(args.target)

    if not tests:
        print(f"No tests found in '{args.target}' with a '{CONFIG_START}' block.")
        sys.exit(0)

    filtered = filter_tests(tests, args.importance, args.tag)

    if not filtered:
        print(f"No tests found matching the criteria in '{args.target}'.")
        sys.exit(0)

    if args.interactive:
        global_config = load_global_config()
        run_interactive_mode(filtered, global_config, args.dry_run)
        sys.exit(0)

    if args.action == "list":
        print(f"\nDiscovered {len(filtered)} test(s) in '{args.target}':")
        print_tests_table(filtered)
        print("\nUse 'python scripts/run_tests.py run <path>' to execute them, or use -i for interactive mode.\n")
    elif args.action == "run":
        global_config = load_global_config()
        if global_config:
            print("Loaded global config from test_config.yaml")

        for filepath, config, importance, tags in filtered:
            run_test(filepath, args.run_level, args.job, global_config, args.dry_run)
