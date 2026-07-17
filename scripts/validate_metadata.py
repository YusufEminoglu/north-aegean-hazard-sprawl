"""Validate citation metadata, parameter tables, and local site links."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def no_doi(value, location="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() == "doi":
                raise AssertionError(f"DOI field must remain absent before acceptance: {location}")
            no_doi(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            no_doi(child, f"{location}[{index}]")


def validate_citation():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["license"] == "MIT"
    assert len(citation["authors"]) == 2
    assert "under review" in citation["preferred-citation"]["notes"].lower()
    no_doi(citation)


def validate_weights():
    path = ROOT / "data" / "ahp" / "weight_schemes.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    numeric_columns = [name for name in rows[0] if name != "criterion"]
    for column in numeric_columns:
        total = sum(float(row[column]) for row in rows)
        assert abs(total - 1.0) <= 0.011, f"{column} sums to {total}"


def resolve_site_link(link):
    clean = link.split("#", 1)[0].split("?", 1)[0]
    if not clean or clean.startswith(("http://", "https://", "mailto:", "data:", "#")):
        return None
    if clean.startswith("figures/"):
        return ROOT / clean
    if clean in {"CITATION.cff", "LICENSE", "LICENSE-docs", "NOTICE.md"}:
        return ROOT / clean
    return ROOT / "docs" / clean


def validate_site_links():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    links = re.findall(r"""(?:href|src)=["']([^"']+)["']""", html)
    missing = []
    for link in links:
        target = resolve_site_link(link)
        if target is not None and not target.exists():
            missing.append(f"{link} -> {target.relative_to(ROOT)}")
    assert not missing, "Missing local site links:\n  " + "\n  ".join(missing)


def validate_readme_links():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", readme)
    missing = []
    for link in links:
        clean = link.split("#", 1)[0].split("?", 1)[0]
        if not clean or clean.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = ROOT / clean
        if not target.exists():
            missing.append(clean)
    assert not missing, "Missing README links:\n  " + "\n  ".join(missing)


def main():
    validate_citation()
    validate_weights()
    validate_site_links()
    validate_readme_links()
    print("Metadata validation passed.")


if __name__ == "__main__":
    main()
