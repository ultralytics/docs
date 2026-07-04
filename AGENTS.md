# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
uv pip install -r requirements.txt                     # beautifulsoup4, requests, pandas (for utils/)
python3 utils/check_image_sizes.py <download_dir> <website>  # flag images >750 KB, as links.yml runs it
lychee --scheme 'https' './**/*.md' './**/*.html'      # PR link check (simplified); CI adds more flags, see .github/workflows/links_local.yml
npx prettier --write "**/*.md" "**/*.yml"              # Markdown/YAML formatting
codespell docs utils README.md                          # spelling
```

- There is no test suite, build, or coverage — PR CI is `links_local.yml` (lychee over all repo `*.md`/`*.html` against the live web, so a dead URL fails CI) plus Ultralytics Actions formatting in `format.yml` (source of truth for Prettier/Ruff/docformatter/codespell settings; it runs them server-side on PRs).
- Workflows run on `ubuntu-latest` with unpinned Python (`3.x`; `check_domains.yml` uses 3.14); no language floor applies to this repo itself.

## Architecture

This repo sources the live https://docs.ultralytics.com/compare/ pages: 156 pairwise model-comparison Markdown files plus an `index.md` hub in `docs/en/compare/` (the only doc content here) plus website QA automation. The main docs (models, tasks, guides) live in the `ultralytics/ultralytics` repo under `docs/en/`, and the two trees are merged at site build time — relative links like `../models/yolo26.md` in compare pages resolve against the ultralytics repo, so they appear broken locally but are not. There is no mkdocs config here; the site is built outside this repo's main branch, with built HTML landing on the `gh-pages` branch, and `sitemaps.yml` triggers on successful `pages-build-deployment` runs to submit sitemaps to Google Search Console and changed URLs to IndexNow.

The remaining workflows handle website QA and housekeeping: `links.yml` (daily 07:00 UTC, downloads rendered www/docs/handbook sites and checks links with lychee, spelling with codespell, and image sizes with `utils/check_image_sizes.py`, alerting Slack on failures), `links_local.yml` (repo link check on push/PR to `main`/`gh-pages` and daily 00:00 UTC), `check_domains.yml` (daily redirect checks across ~36 Ultralytics domains), `download_websites.yml` (manual-only site download), and `stale.yml` (issue/PR staleness, not website QA). Releases are manual: `tag.yml` is `workflow_dispatch`-only and gated to `github.repository == 'ultralytics/docs' && github.actor == 'glenn-jocher'`; there is no version file or package publish.

## Conventions

- Every `.py`/`.yml` file opens with the `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header — Ultralytics Actions adds it automatically, so don't add or revert these manually.
- Ultralytics Actions pushes auto-format commits directly to PR branches; always `git pull --rebase` before pushing.
- Pairwise compare pages follow a fixed shape: YAML frontmatter (`title`, `comments: true`, `description`, `keywords` — the `index.md` hub omits `title`), a Chart.js `<canvas>` fed by `benchmark.js`, and benchmark tables where **bold** marks the better value; "Learn more" buttons link to platform.ultralytics.com only for models with Platform pages (YOLO26, YOLO11, YOLOv8, YOLOv5) and to docs.ultralytics.com or GitHub for the rest.
- Link-checker exclusions live in `.lycheeignore` (one regex per line) and in the `--exclude` lists inside `links.yml`/`links_local.yml`; the bot-protected-domain regex is duplicated verbatim in both workflows and should stay in sync, while the other `--exclude` patterns and `--accept` codes are intentionally workflow-specific.
- All CI checks hit the live network by design (link checks, domain redirects, sitemap submission); expect occasional flakes from bot-protected domains, handled via `ultralytics/actions/retry` wrappers plus the accept-code and exclude lists.
