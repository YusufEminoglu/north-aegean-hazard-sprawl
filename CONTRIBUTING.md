# Contributing

Contributions that improve reproducibility, portability, documentation, or
validation are welcome.

1. Open an issue describing the scientific or technical change.
2. Create a focused branch and keep third-party data out of Git.
3. Run the publication audit, the AHP check, and Python compilation.
4. Explain whether the change affects any reported statistic or figure.
5. Submit a pull request using the repository template.

Required local checks:

    python scripts/publication_audit.py
    python scripts/verify_ahp.py
    python -m compileall -q src scripts

Do not submit credentials, Earth Engine project identifiers, local user paths,
raw geospatial data, manuscript files, or publisher PDFs. Scientific changes
should include their assumptions, parameter values, and expected effects.
