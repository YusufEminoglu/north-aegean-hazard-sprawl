"""Fail fast on files and strings that must never enter the public repository."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "LICENSE",
    "LICENSE-docs",
    "CITATION.cff",
    ".zenodo.json",
    "docs/data_sources.md",
    "docs/methods.md",
    "docs/ahp_matrices.md",
}
FORBIDDEN_SUFFIXES = {
    ".tif", ".tiff", ".gpkg", ".shp", ".dbf", ".pkl", ".joblib",
    ".pdf", ".docx", ".tex", ".zip", ".p12", ".pem",
}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {
    "", ".py", ".md", ".txt", ".csv", ".cff", ".yml", ".yaml",
    ".toml", ".json", ".html", ".css", ".js", ".svg", ".example",
}
CHECKS = {
    "absolute user path": re.compile(
        r"[A-Za-z]:[\\/](?:Users|Documents and Settings|Downloads)[\\/]",
        re.IGNORECASE,
    ),
    "hard-coded EE project": re.compile(
        r"ee\.Initialize\(\s*project\s*=\s*['\"]",
        re.IGNORECASE,
    ),
    "hard-coded Drive folder": re.compile(
        r"folder\s*=\s*['\"][^'\"]+['\"]",
        re.IGNORECASE,
    ),
    "private EE asset": re.compile(
        r"projects/(?!glad/|sat-io/|soilgrids-isric/)[\w.-]+/assets/",
        re.IGNORECASE,
    ),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
}


def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main() -> None:
    failures: list[str] = []

    for required in REQUIRED:
        if not (ROOT / required).exists():
            failures.append(f"missing required file: {required}")

    for path in iter_files():
        relative = path.relative_to(ROOT)
        suffix = path.suffix.lower()

        if suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden publishable artifact: {relative}")
            continue
        if suffix not in TEXT_SUFFIXES:
            continue
        if relative.as_posix() == "scripts/publication_audit.py":
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for label, pattern in CHECKS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{label}: {relative}:{line}")

    pngs = list((ROOT / "figures" / "png").glob("fig*.png"))
    if len(pngs) != 10:
        failures.append(f"expected 10 authored figures, found {len(pngs)}")

    if failures:
        print("Publication audit failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Publication audit passed: no prohibited artifacts or identifiers found.")


if __name__ == "__main__":
    main()
