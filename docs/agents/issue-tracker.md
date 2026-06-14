# Issue tracker: GitHub (ertelendi)

> **Not (2026-06-09):** GitHub remote henüz kesin değil (şirket vs kişisel karışıklığı).  
> Issue'lar şimdilik **yerel iş paketlerinde**: `docs/agents/work_packages/`

Issues for this repo will live in GitHub Issues on `mUsaulug/CardioGuard-AI` when remote is configured. Use the `gh` CLI for all operations — run from inside the repo and `gh` picks up the remote automatically.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`
- **View an issue**: `gh issue view <number> --comments`
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

## When a skill says "publish to the issue tracker"

Run `gh issue create` with a title and body.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
