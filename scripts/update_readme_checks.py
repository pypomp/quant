"""Append each report's worst automated verdict to its README link, e.g.
`[report](tests/spx/estimation/report.html) · PASS`. A link into a cross-model
report, `[report](reports/loglik.html#spx)`, gets that model's verdict from
reports/loglik_checks.json. Reports whose checks are all info get no verdict."""

import json
import os
import re

README = "README.md"
LINK = re.compile(
    r"(\[[^\]]+\]\(((?:tests|reports)/[^)#]+)\.html(?:#([\w-]+))?\))"
    r"(?: · (?:PASS|CHECK|FAIL|n/a))?"
)


def verdict(page, model):
    if page.startswith("reports/"):
        path = f"{page}_checks.json"
    else:
        path = os.path.join(os.path.dirname(page), "checks.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        checks = json.load(f)
    if "models" in checks:
        checks = checks["models"].get(model, {})
    return checks.get("verdict")


def annotate(match):
    v = verdict(match.group(2), match.group(3))
    return match.group(1) if v in (None, "info") else f"{match.group(1)} · {v}"


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
