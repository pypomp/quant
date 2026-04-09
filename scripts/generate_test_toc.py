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
            if file == "report.html":
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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Outfit:wght@600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f172a;
            --card: #1e293b;
            --primary: #6366f1;
            --hover: #242f44;
            --text: #f1f5f9;
            --muted: #94a3b8;
        }}
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            padding: 5rem 2rem;
            min-height: 100vh;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        header {{ margin-bottom: 4rem; border-left: 5px solid var(--primary); padding-left: 2rem; }}
        h1 {{ font-family: 'Outfit', sans-serif; font-size: 3.5rem; letter-spacing: -0.04em; margin-bottom: 0.5rem; }}
        .subtitle {{ color: var(--muted); font-size: 1.25rem; font-weight: 300; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 2rem; }}
        .card {{
            background: var(--card);
            border-radius: 20px;
            padding: 2.5rem;
            border: 1px solid #334155;
            text-decoration: none;
            color: inherit;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: block; box-shadow: 0 4px 6px -1px #000;
        }}
        .card:hover {{ transform: translateY(-10px); border-color: var(--primary); background: var(--hover); box-shadow: 0 25px 30px -10px #000; }}
        .tag {{ color: var(--primary); font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.1em; display: block; margin-bottom: 1.5rem; }}
        h2 {{ font-family: 'Outfit', sans-serif; font-size: 1.75rem; margin-bottom: 1.5rem; line-height: 1.1; }}
        .cta {{ display: flex; align-items: center; font-weight: 600; color: #818cf8; font-size: 0.95rem; }}
        .cta svg {{ width: 18px; height: 18px; margin-left: 0.5rem; transition: transform 0.2s; }}
        .card:hover .cta svg {{ transform: translateX(6px); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Research Library</h1>
            <p class="subtitle">Research output across the quant repository.</p>
        </header>

        <main class="grid">
            {
        "".join(
            [
                f'''
            <a href="{r['path']}" class="card">
                <span class="tag">{r['folder']}</span>
                <h2>{r['title']}</h2>
                <div class="cta">
                    Open Report
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"></path></svg>
                </div>
            </a>
            '''
                for r in reports
            ]
        )
    }
        </main>
    </div>
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
