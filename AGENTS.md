# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

This repository (AGPL-3.0) holds the pairwise model-comparison pages published at https://docs.ultralytics.com/compare/ plus the automated QA workflows that check links, spelling, sitemaps, and image sizes across the Ultralytics websites. The rest of the documentation lives in `ultralytics/ultralytics` under `docs/en/`.

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

This repo sources the live https://docs.ultralytics.com/compare/ pages: 156 pairwise model-comparison Markdown files plus an `index.md` hub in `docs/en/compare/` (the only doc content here) plus website QA automation. The main docs (models, tasks, guides) live in the `ultralytics/ultralytics` repo under `docs/en/`, and the two trees are merged at site build time — relative links like `../models/yolo26.md` in compare pages resolve against the ultralytics repo, so they appear broken locally but are not. There is no mkdocs config here; the site is built and deployed from the private `ultralytics/portal` repository, which also centralizes IndexNow submission. Portal clones this repo at build time, so a merge here only reaches the live site when `publish.yml` fires the Vercel deploy hook. The legacy `gh-pages` branch and its GitHub Pages site were deleted on 2026-08-09; nothing publishes to that branch any more.

`publish.yml` redeploys docs.ultralytics.com: it POSTs the Vercel deploy hook on every push to `main`, plus a daily 05:00 UTC run as a backstop against hook or path-filter drift here and in `ultralytics/ultralytics`. The remaining workflows handle docs-specific website QA and housekeeping: `links.yml` (daily 07:00 UTC, downloads rendered www/docs/academy/handbook sites and checks links with lychee, spelling with codespell, and image sizes with `utils/check_image_sizes.py`, alerting Slack on failures), `links_local.yml` (repo link check on push/PR to `main` and daily 00:00 UTC), `download_websites.yml` (manual-only site download), and `stale.yml` (issue/PR staleness, not website QA). Cross-site sitemap submission and domain redirect checks are centralized in the private `ultralytics/portal` repository. Releases are manual: `tag.yml` is `workflow_dispatch`-only and gated to `github.repository == 'ultralytics/docs' && github.actor == 'glenn-jocher'`; there is no version file or package publish.

## Conventions

- Every `.py`/`.yml` file opens with the `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header — Ultralytics Actions adds it automatically, so don't add or revert these manually.
- Pairwise compare pages follow a fixed shape: YAML frontmatter (`title`, `comments: true`, `description`, `keywords` — the `index.md` hub omits `title`), a Chart.js `<canvas>` fed by `benchmark.js`, and benchmark tables where **bold** marks the better value; model `{ .md-button }` links (usually "Learn more about <model>") point to platform.ultralytics.com only for models with Platform pages (YOLO26, YOLO11, YOLOv8, YOLOv5) and to docs.ultralytics.com or GitHub for the rest.
- Link-checker exclusions live in `.lycheeignore` (one regex per line) and in the `--exclude` lists inside `links.yml`/`links_local.yml`; the bot-protected-domain regex is duplicated verbatim in both workflows and should stay in sync, while the other `--exclude` patterns and `--accept` codes are intentionally workflow-specific.
- All CI checks hit the live network by design (link checks, domain redirects, sitemap submission); expect occasional flakes from bot-protected domains, handled via `ultralytics/actions/retry` wrappers plus the accept-code and exclude lists.
