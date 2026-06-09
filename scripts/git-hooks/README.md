# Git hooks

Enable once per clone:

```bash
git config core.hooksPath scripts/git-hooks
```

`prepare-commit-msg` removes `Co-authored-by` trailers so bot accounts are not listed as GitHub contributors.
