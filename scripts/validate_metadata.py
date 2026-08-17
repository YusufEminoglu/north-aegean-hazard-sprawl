"""Validate citation metadata, author identities, parameter tables, and links."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/YusufEminoglu/north-aegean-hazard-sprawl"
ORCIDS = {
    "Yusuf Eminoğlu": "https://orcid.org/0009-0005-6000-2934",
    "Kemal Mert Çubukçu": "https://orcid.org/0000-0003-3604-7014",
}
EXPECTED_FIGURES = {
    "fig01_Baseline.png",
    "fig02_Dynamics.png",
    "fig03_Drivers.png",
    "fig04_Suitability.png",
    "fig05_CA_Scenarios.png",
    "fig06_MultiHazard_Convergence.png",
    "fig07_Sensitivity.png",
    "fig08_Exposure.png",
    "fig09_Demographic_Exposure.png",
    "fig10_Policy_Synthesis.png",
    "fig11_Prioritization.png",
    "fig12_WildfireSubcomponents.png",
    "fig13_LocationMap.png",
}


def no_doi(value, location="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() == "doi":
                raise AssertionError(
                    f"DOI field must remain absent before acceptance: {location}"
                )
            no_doi(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            no_doi(child, f"{location}[{index}]")


def full_name(author):
    return f"{author['given-names']} {author['family-names']}"


def validate_citation():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["license"] == "MIT"
    assert citation["repository-code"] == REPOSITORY_URL
    assert len(citation["authors"]) == 2
    actual = {full_name(author): author["orcid"] for author in citation["authors"]}
    assert actual == ORCIDS
    preferred = citation["preferred-citation"]
    assert "under review" in preferred["notes"].lower()
    preferred_orcids = {
        full_name(author): author["orcid"] for author in preferred["authors"]
    }
    assert preferred_orcids == ORCIDS
    no_doi(citation)


def validate_zenodo():
    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert metadata["license"] == "MIT"
    assert metadata["upload_type"] == "software"
    assert metadata["access_right"] == "open"
    actual = {
        creator["name"]: "https://orcid.org/" + creator["orcid"]
        for creator in metadata["creators"]
    }
    expected = {
        "Eminoğlu, Yusuf": ORCIDS["Yusuf Eminoğlu"],
        "Çubukçu, Kemal Mert": ORCIDS["Kemal Mert Çubukçu"],
    }
    assert actual == expected
    no_doi(metadata)


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
        return ROOT / (".zenodo.json" if clean == "zenodo.json" else clean)
    if clean in {
        "CITATION.cff", ".zenodo.json", "zenodo.json", "LICENSE", "LICENSE-docs", "NOTICE.md"
    }:
        return ROOT / (".zenodo.json" if clean == "zenodo.json" else clean)
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
    for name, orcid in ORCIDS.items():
        assert name in readme
        assert orcid in readme


def validate_figures():
    actual = {path.name for path in (ROOT / "figures" / "png").glob("*.png")}
    assert actual == EXPECTED_FIGURES


def main():
    validate_citation()
    validate_zenodo()
    validate_weights()
    validate_site_links()
    validate_readme_links()
    validate_figures()
    print("Metadata validation passed: authors, ORCIDs, archive metadata, and links.")


if __name__ == "__main__":
    main()
