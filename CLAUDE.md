# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Harpreet Matharoo's personal site and blog, built with [Quarto](https://quarto.org) and served at `https://mathadoor.github.io` (a GitHub user page, no custom domain). Previously a Jekyll site on the al-folio theme; migrated to Quarto to cut down on theme/plugin machinery.

## Commands

```bash
quarto preview   # render + serve locally with live reload
quarto render    # build the site to _site/ without serving
```

No test suite. `.pre-commit-config.yaml` runs basic hygiene hooks (trailing whitespace, end-of-file, YAML check, large files): `pre-commit run --all-files`.

## Structure

- `_quarto.yml` — site config: navbar, site-wide giscus comment settings, HTML theme.
- `index.qmd` — the about/home page (`about: template: trestles`), rendered at `/`.
- `blog/index.qmd` — the blog listing page (`listing:` over `blog/*/*/index.qmd`), rendered at `/blog/`.
- `blog/<year>/<slug>/index.qmd` — individual posts, co-located with any images they use. **The `<year>/<slug>` path is deliberately kept identical to the old Jekyll `/blog/:year/:title/` permalinks** so existing links and search results keep working — don't rename an existing post directory without a reason.
- `data/cv.yml`, `data/repositories.yml` — real CV/repository data carried over from the Jekyll site, not currently wired into any page (no CV or repositories page exists yet).

### Redirect posts

Some posts (e.g. `blog/2023/mind-projection/`) have no real content — they're just an external link (to a Medium post, an app, another site). These are implemented as a normal post (so they still appear in the blog listing with title/date/description) whose body is a `` ```{=html}<meta http-equiv="refresh" ...>``` `` block that bounces the visitor to the external URL on load, with `comments: false` in the front matter. Follow this pattern for any new link-out post.

### A known, deliberately preserved quirk

`blog/2023/transformer-expo-copy/` actually contains the "Recognizing Handwritten Mathematical Expressions" post, and `blog/2023/Recognizing-HME/` actually contains the "Transformer Exposition" post — the directory names and content are swapped. This mismatch already existed on the live Jekyll site; it was preserved as-is during the Quarto migration rather than fixed, to avoid changing live URLs. Don't "fix" this without checking with the site owner first.

## Content conventions

- Front matter: `title`, `description`, `date` (YYYY-MM-DD), `categories: [Tag1, Tag2]` as a YAML list (the old Jekyll site had separate `tags`/`categories` fields; these have been consolidated into one `categories` list per post).
- Comments are giscus, configured once at the site level in `_quarto.yml`; override per-post with `comments: false` when a post shouldn't have a comment thread (used by the redirect posts above).
- Math: standard Pandoc conventions — `$...$` for inline, `$$...$$` for display blocks only (the old Jekyll/kramdown posts used `$$...$$` for inline math too; this was corrected during migration).
- Images live alongside the post that uses them (`blog/<year>/<slug>/image.png`), referenced with a plain relative Markdown image and a caption in the alt text — no more `{% include figure.html %}`.

## Deployment

`.github/workflows/deploy.yml` renders the site and publishes `_site/` to the `gh-pages` branch via `quarto-dev/quarto-actions/publish`, triggered on push to `master`. GitHub Pages is configured to serve from the `gh-pages` branch (this predates the Quarto migration and hasn't changed).
