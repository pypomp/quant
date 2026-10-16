"""Append each report's worst automated verdict (from its checks.json) to its
README link, e.g. `[report](tests/spx/loglik/report.html) · PASS`. Reports
whose checks are all info get no verdict."""

import json
import os
import re

README = "README.md"
LINK = re.compile(
    r"(\[[^\]]+\]\((tests/[^)]+)/[\w-]+\.html\))(?: · (?:PASS|CHECK|FAIL|n/a))?"
)


def annotate(match):
    path = os.path.join(match.group(2), "checks.json")
    if not os.path.exists(path):
        return match.group(1)
    with open(path) as f:
        verdict = json.load(f)["verdict"]
    return match.group(1) if verdict == "info" else f"{match.group(1)} · {verdict}"


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
