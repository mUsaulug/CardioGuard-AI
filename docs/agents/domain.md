# Domain Docs

How the engineering skills should consume this repo's domain documentation.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — domain glossary and bounded context.
- **`docs/adr/`** — architectural decision records. Read ADRs that touch the area you're about to work in.

If either doesn't exist yet, proceed silently. They are created lazily via `/grill-with-docs` as terms and decisions crystallise.

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── docs/
│   ├── adr/
│   │   └── 0001-*.md
│   └── agents/       ← this directory
└── src/
```

## Use the glossary's vocabulary

When naming things in issues, refactor proposals, test names, or hypotheses — use terms as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If a concept you need isn't in the glossary, either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0002 (ensemble formula) — but worth reopening because…_
