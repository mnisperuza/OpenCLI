# Releasing Fenrir Agent

Fenrir Agent releases are immutable version tags. The release workflow verifies the
tag, runs the full suite, builds the source distribution and wheel, validates
metadata, installs the built wheel in a clean environment, writes SHA-256
checksums, creates GitHub Release assets, and then publishes to PyPI.

## One-time repository setup

1. In PyPI, create or claim the `fenrir-agent` project under the intended owner.
2. Configure PyPI Trusted Publishing for this GitHub repository, workflow file
   `.github/workflows/release.yml`, and the `pypi` GitHub environment.
3. In GitHub, create the protected `pypi` environment. Require approval if a
   maintainer should explicitly authorize public publication.
4. Protect version tags and require the `Harness release gates` workflow on
   changes that will be released.

No API token is stored in GitHub. PyPI Trusted Publishing exchanges the
workflow's short-lived GitHub identity token for publication authority.

## Release procedure

1. Update `fenrir_agent/_version.py` and move completed items into `CHANGELOG.md`.
2. Run `python -m pytest -q`, `python -m build`, `python -m twine check dist/*`,
   and `python scripts/package_smoke.py dist` locally.
3. Commit the release changes and create an annotated tag matching the version:

   ```bash
   git tag -a v2.0.0 -m "Fenrir Agent 2.0.0"
   git push origin v2.0.0
   ```

4. Approve the `pypi` environment when GitHub asks. Verify the GitHub Release,
   checksums, and PyPI project page after the workflow completes.

If a package has already been uploaded to PyPI, do not reuse its version. Make a
new patch release and tag that new version instead.
