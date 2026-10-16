"""Append each report's worst automated verdict (from its checks.json) to its
README link, e.g. `[report](tests/spx/loglik/report.html) · PASS`."""

import json
import os
import re

README = "README.md"
LINK = re.compile(
    r"(\[report\]\((tests/[^)]+)/report\.html\))(?: · (?:PASS|CHECK|FAIL|n/a))?"
)


def annotate(match):
    path = os.path.join(match.group(2), "checks.json")
    if not os.path.exists(path):
        return match.group(1)
    with open(path) as f:
        return f"{match.group(1)} · {json.load(f)['verdict']}"


def main():
    with open(README) as f:
        text = f.read()
    new = LINK.sub(annotate, text)
    if new != text:
        with open(README, "w") as f:
            f.write(new)
    print("README.md updated" if new != text else "README.md unchanged")


if __name__ == "__main__":
    main()
