# mathadoor.github.io

Harpreet Matharoo's personal site and blog ("Hermes"), built with [Quarto](https://quarto.org) and served at [mathadoor.github.io](https://mathadoor.github.io).

## Local development

```bash
quarto preview
```

Renders the site to `_site/` and serves it locally with live reload.

```bash
quarto render
```

Builds the site to `_site/` without serving it.

## Deployment

Pushing to `master` triggers `.github/workflows/deploy.yml`, which renders the site and publishes `_site/` to the `gh-pages` branch via [quarto-actions](https://github.com/quarto-dev/quarto-actions).
