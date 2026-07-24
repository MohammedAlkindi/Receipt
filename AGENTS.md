# AGENTS.md — Receipt

Read this and `CLAUDE.md` before working here. Full standards: `~/.claude/CLAUDE.md` and `~/Github/CLAUDE.md`.

## Scope
Implement features, fix bugs, and refactor within this repo.

## Branch & commits
- Branch prefix by agent: Claude → `claude/<type>-<kebab>`, Codex → `Codex/<type>-<kebab>`.
- Conventional-commit messages; one logical change per commit; PRs via the `gh` CLI.

## Requires explicit confirmation
Destructive git ops (force-push, history rewrite), deploys/publishing, secret or CI changes, and data deletion.

## Before done
Run the test suite (command in `CLAUDE.md`) and confirm it passes.

