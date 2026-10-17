# Releasing FreeCode

## One-time setup

1. **GitHub repo** — create an empty repo and push:

   ```bash
   git remote add origin git@github.com:<you>/FreeCode.git
   git push -u origin main
   ```

2. **PyPI Trusted Publishing** — on [pypi.org → Publishing](https://pypi.org/manage/account/publishing/)
   add a *pending publisher*:

   | field | value |
   | --- | --- |
   | Project name | `freecode-ai` |
   | Owner | your GitHub user/org |
   | Repository | `FreeCode` |
   | Workflow name | `release.yml` |
   | Environment | *(leave empty)* |

   No API tokens or passwords — GitHub Actions proves who it is with an
   OIDC token, so nothing secret ever lives in the repo.

## Every release

1. Bump the version in **both** places (they must match):
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `freecode/__init__.py` → `__version__ = "X.Y.Z"`
2. Commit, tag, push:

   ```bash
   git add pyproject.toml freecode/__init__.py
   git commit -m "vX.Y.Z"
   git tag vX.Y.Z
   git push origin main --tags
   ```

3. The **Release** workflow builds the sdist + wheel, runs `twine check`,
   and publishes to PyPI automatically. Watch it under the repo's Actions tab.
4. Verify: `pip install freecode-ai==X.Y.Z` in a clean venv.

Note: PyPI doesn't allow re-uploading an existing version — always bump the
version before tagging.
