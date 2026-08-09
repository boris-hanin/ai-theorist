# Contributing

This repository is a research record. Preserve failed predictions, raw results,
and provenance; never rewrite a round to make a bar pass.

Before committing, verify that Git is using your identity rather than a
placeholder:

```bash
git config user.name
git config user.email
```

For a new round:

1. Copy `rounds/TEMPLATE/prereg.md`, complete it, and commit it before running.
2. Commit runner code before launching an expensive job.
3. Save raw machine-readable results beside the human report.
4. Use shared seeds for paired comparisons and report the paired uncertainty.
5. Run `pytest` and the relevant validation scripts before opening a change.

Do not rewrite existing Git history to repair old placeholder authors. A
history rewrite changes every downstream commit identifier and requires an
explicit repository-owner decision.
