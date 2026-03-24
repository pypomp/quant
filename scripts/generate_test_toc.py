import os
import re
import sys
import argparse
from pathlib import Path
import yaml
import ast
import subprocess

CONFIG_START = "--- SLURM CONFIG ---"
CONFIG_END = "--- END SLURM CONFIG ---"

def parse_test_metadata(filepath):
    """Extracts and parses the YAML configuration from comments in the test script."""
    yaml_lines = []
    in_config = False
    
    with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Both R and Python use '#' for comments
            if not stripped.startswith("#"):
                # If we hit actual code, keep looking (unless we are inside config)
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
        yaml_str = "".join(yaml_lines)
        config = yaml.safe_load(yaml_str)
        return config if config else {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in {filepath}: {e}")
        return None

def get_python_docstring(filepath):
    """Extracts the module-level docstring from a Python file."""
    try:
        with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
            tree = ast.parse(f.read())
            return ast.get_docstring(tree)
    except Exception:
        return None

def get_r_description(filepath):
    """Extracts description from R file using #' (Roxygen) or first comment block."""
    description_lines = []
    found_roxygen = False
    
    with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#'"):
                found_roxygen = True
                content = stripped[2:].strip()
                # Remove trailing #' if it was used for wrapping on the same line
                if content.endswith("#'"):
                    content = content[:-2].strip()
                description_lines.append(content)
            elif not found_roxygen and stripped.startswith("#"):
                # If we haven't found Roxygen, collect normal comments at the top
                # but stop at SLURM CONFIG
                if CONFIG_START in line:
                    break
                content = stripped[1:].strip()
                if content and not content.startswith("-"): # avoid headers like ## ---
                    description_lines.append(content)
            elif not stripped:
                if description_lines: # Stop at first empty line after a comment block
                    break
                continue
            else:
                # Actual code or end of leading comment block
                break
    
    return "\n".join(description_lines).strip() if description_lines else None

def find_tests(target_path):
    target_path = Path(target_path)
    test_files = []
    
    for ext in ["*.py", "*.R", "*.r"]:
        for file in target_path.rglob(ext):
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(4000) # Read enough to find config
                    if CONFIG_START in content:
                        test_files.append(file)
            except Exception:
                continue
    return sorted(test_files)

def generate_toc(target_dir, output_file, render=True):
    tests = find_tests(target_dir)
    if not tests:
        print(f"No tests found in {target_dir}")
        return

    project_root = Path(target_dir).absolute()
    
    # Organize tests by directory structure
    tree = {}
    for test_path in tests:
        test_path_abs = test_path.absolute()
        rel_path = test_path_abs.relative_to(project_root)
        parts = rel_path.parts[:-1]
        
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        if "__files__" not in current:
            current["__files__"] = []
        current["__files__"].append(test_path)

    toc_lines = ["---"]
    toc_lines.append("title: Table of Contents for Tests")
    toc_lines.append("toc: true")
    toc_lines.append("format:")
    toc_lines.append("  html:")
    toc_lines.append("    embed-resources: true")
    toc_lines.append("---\n")
    toc_lines.append("This document is automatically generated. It lists all available tests in the `tests/` directory.\n")

    def walk_tree(node, depth=1):
        # First process subdirectories (alphabetical)
        for folder_name in sorted(k for k in node.keys() if k != "__files__"):
            header = "#" * (depth + 1)
            toc_lines.append(f"{header} {folder_name}\n")
            walk_tree(node[folder_name], depth + 1)
        
        # Then process files in this directory
        if "__files__" in node:
            file_header = "#" * (depth + 1)
            for test_path in sorted(node["__files__"]):
                test_path_abs = test_path.absolute()
                rel_path = test_path_abs.relative_to(project_root)
                config = parse_test_metadata(test_path)
                
                # Extract description
                description = None
                if test_path.suffix == ".py":
                    description = get_python_docstring(test_path)
                elif test_path.suffix in [".R", ".r"]:
                    description = get_r_description(test_path)
                
                # Extract jobs
                job_names = []
                if isinstance(config, dict):
                    if "jobs" in config and isinstance(config["jobs"], dict):
                        job_names = list(config["jobs"].keys())
                    else:
                        job_names = [config.get("name", test_path.name)]
                elif config is not None:
                    print(f"Warning: Unexpected config type {type(config)} in {rel_path}")
                
                # Write to TOC
                toc_lines.append(f"{file_header}# `{rel_path.name}`\n")
                toc_lines.append(f"- **Path**: `{rel_path}`\n")
                if job_names:
                    toc_lines.append(f"- **Jobs**: {', '.join(f'`{j}`' for j in job_names)}\n")
                
                if description:
                    # Clean up description (first paragraph or whole thing)
                    desc_body = description.strip()
                    toc_lines.append(f"\n{desc_body}\n")
                
                toc_lines.append("\n---\n")

    walk_tree(tree)

    with open(output_file, "w", encoding='utf-8') as f:
        f.write("\n".join(toc_lines))
    
    print(f"Generated TOC at {output_file}")
    
    if render:
        print(f"Rendering {output_file} to HTML via Quarto...")
        try:
            # Capture output so we can see what's happening if it fails
            result = subprocess.run(
                ["quarto", "render", output_file], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print("Rendering complete.")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Quarto rendering failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
        except FileNotFoundError:
            print("Warning: 'quarto' command not found in PATH. Skipping HTML rendering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Markdown TOC for tests")
    parser.add_argument("target", nargs="?", default="tests", help="Directory containing tests")
    parser.add_argument("--output", default="TESTS_INDEX.md", help="Output Markdown file")
    parser.add_argument("--no-render", action="store_false", dest="render", help="Skip Quarto rendering")
    
    args = parser.parse_args()
    
    generate_toc(args.target, args.output, args.render)
