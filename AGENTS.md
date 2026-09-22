# FinsSimIsaacLab Agent Guide

## Release branches and remotes

- `origin-private` is the private development remote. Push its `main`, the
  single moving `release-candidate`, and internal `release/<version>` branches here.
- `origin` is the public release remote. Its `main` always represents the
  current history-free release snapshot; never push private `main` or a
  candidate branch there.
- Private `main` retains full development history. Create one
  `release-candidate` branch from it to remove incomplete features, local
  assets, experiments, and other material not approved for publication.
- `scripts/publish_public_release.sh` creates `release/<version>` as an orphan
  root commit from the committed candidate tree, updates `origin/main` using
  `--force-with-lease`, and publishes an immutable annotated `v<version>` tag.
- Do not merge a release back into private main or rewrite a published tag in normal operation. Only an approved, low-risk corrective reissue may use `--amend-release` to rebuild the same version, update `origin/main`, and rewrite its tag; record the reason in the commit/release notes. New features or behavior changes require a new version.
- Mark candidate revisions with immutable annotated tags such as
  `v0.1.0-rc.1`, rather than encoding the version in the branch name.

```bash
git push origin-private main
git push origin-private release-candidate
scripts/publish_public_release.sh --version 0.1.0 \
  --exclude-file path/to/release-excludes.txt
```

Keep generated caches, Isaac build products, checkpoints, recordings, and
unreviewed third-party assets out of release snapshots.
