import os
import re
import subprocess
from pathlib import Path

# Directories to exclude from the search (in addition to .gitignore)
# Hidden directories (starting with '.') are ignored by default.
DEFAULT_EXCLUDE = {"renv", "node_modules", "__pycache__"}

# Add your custom ignored directories here (e.g., {"archive", "temp"})
USER_EXCLUDE = set({"daphnia"})


def get_html_title(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read(8192)
            match = re.search(
                r"<title>(.*?)</title>", content, re.IGNORECASE | re.DOTALL
            )
            if match:
                return match.group(1).strip()
    except Exception:
        pass
    return None


def is_ignored(path):
    # Returns 0 if ignored, 1 if not ignored
    try:
        subprocess.check_call(["git", "check-ignore", "-q", str(path)])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    root_dir = Path(".")
    reports = []

    # Combine default and user exclusions
    all_exclude = DEFAULT_EXCLUDE.union(USER_EXCLUDE)

    for root, dirs, files in os.walk(root_dir):
        # Filter directories: skip hidden ones and those in our exclusion list
        dirs[:] = [d for d in dirs if d not in all_exclude and not d.startswith(".")]

        for file in files:
            if file in ("report.html", "test.html"):
                path = Path(root) / file
                if is_ignored(path):
                    continue
                title = get_html_title(path)
                rel_path = str(path.relative_to(root_dir))
                folder = str(path.parent.relative_to(root_dir))
                reports.append(
                    {
                        "title": title or folder.replace("/", " / ").title(),
                        "path": rel_path,
                        "folder": folder,
                    }
                )

    reports.sort(key=lambda x: x["folder"])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Portfolio</title>
    <style>
        body {{
            font-family: sans-serif;
            max-width: 800px;
            margin: 2rem auto;
            padding: 0 1rem;
            line-height: 1.5;
        }}
        ul {{
            padding-left: 1.5rem;
        }}
        li {{
            margin-bottom: 0.75rem;
        }}
        .tag {{
            font-size: 0.85em;
            color: #666;
            margin-left: 0.5rem;
        }}
    </style>
</head>
<body>
    <h1>Research Library</h1>
    <p>Research output across the quant repository.</p>
    <hr>
    <ul>
        {
        "".join(
            [
                f'''
        <li>
            <a href="{r['path']}">{r['title']}</a>
            <span class="tag">({r['folder']})</span>
        </li>'''
                for r in reports
            ]
        )
    }
    </ul>
</body>
</html>"""

    output_path = root_dir / "TESTS_INDEX.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(
        f"Generated index at {output_path} with {len(reports)} items (Git-ignored and manually excluded files skipped)."
    )


if __name__ == "__main__":
    main()
