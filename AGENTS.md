# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

This repository (AGPL-3.0) holds public documentation source consumed by https://docs.ultralytics.com/ plus automated QA workflows that check links, spelling, sitemaps, and image sizes across Ultralytics websites. Additional documentation source lives in `ultralytics/ultralytics` under `docs/en/`.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
uv pip install -r requirements.txt                     # beautifulsoup4, requests, pandas (for utils/)
uv run python utils/check_image_sizes.py <download_dir> <website>  # flag images >750 KB, as links.yml runs it
lychee --scheme 'https' './**/*.md' './**/*.html'      # PR link check (simplified); CI adds more flags, see .github/workflows/links_local.yml
npx prettier --write "**/*.md" "**/*.yml"              # Markdown/YAML formatting
codespell docs utils README.md                          # spelling
```

- There is no test suite, build, or coverage — PR CI is `links_local.yml` (lychee over all repo `*.md`/`*.html` against the live web, so a dead URL fails CI) plus Ultralytics Actions formatting in `format.yml` (source of truth for Prettier/Ruff/docformatter/codespell settings; it runs them server-side on PRs).
- Workflows run on `ubuntu-latest` with unpinned Python (`3.x`); no language floor applies to this repo itself.

## Architecture

This repo is one public content source for https://docs.ultralytics.com/. Additional source lives in `ultralytics/ultralytics` under `docs/en/`; CI combines the sources before running `zensical build --strict`. Relative links may therefore resolve only in the complete content tree. This repository intentionally has no standalone Zensical configuration or site build; the centralized publisher renders and deploys production.

The remaining workflows handle docs-specific publishing, website QA, and housekeeping: `publish.yml` triggers the centralized publisher on every `main` push and daily; `links.yml` downloads rendered www/docs/academy/handbook sites and checks links, spelling, and image sizes; `links_local.yml` checks repository links on push, PR, and daily; `download_websites.yml` is manual-only; and `stale.yml` manages inactive issues and PRs. Releases are manual: `tag.yml` is `workflow_dispatch`-only and gated to `github.repository == 'ultralytics/docs' && github.actor == 'glenn-jocher'`; there is no version file or package publish.

## Conventions

- Every `.py`/`.yml` file opens with the `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header — Ultralytics Actions adds it automatically, so don't add or revert these manually.
- Pairwise compare pages follow a fixed shape: YAML frontmatter (`title`, `comments: true`, `description`, `keywords` — the `index.md` hub omits `title`), a `<canvas>` whose `active-models` data is rendered by the centralized publisher, and benchmark tables where **bold** marks the better value; model `{ .md-button }` links (usually "Learn more about <model>") point to platform.ultralytics.com only for models with Platform pages (YOLO26, YOLO11, YOLOv8, YOLOv5) and to docs.ultralytics.com or GitHub for the rest.
- Link-checker exclusions live in `.lycheeignore` (one regex per line) and in the `--exclude` lists inside `links.yml`/`links_local.yml`; the bot-protected-domain regex is duplicated verbatim in both workflows and should stay in sync, while the other `--exclude` patterns and `--accept` codes are intentionally workflow-specific.
- All CI checks hit the live network by design (link checks, domain redirects, sitemap submission); expect occasional flakes from bot-protected domains, handled via `ultralytics/actions/retry` wrappers plus the accept-code and exclude lists.
