# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hugo static blog ("Salt and Pepper", https://wow2006.github.io/) about computer graphics, vision, and GPU programming. Bilingual: English (default) and Arabic (RTL). Theme: `hello-friend-ng`, a git submodule in `themes/` (the theme is the user's own fork; `MyProductivity` is another submodule). Clone with `--recurse-submodules`.

## Commands

- `hugo server --buildDrafts` — local preview (most new posts are `draft: true`)
- `hugo --buildDrafts` — full build; this is the check that matters (see "fails the build" below)
- Post code samples are standalone CMake projects: `cd assets/code/cuda/postNN && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`. No test suite or linter.

## Architecture

**Content** lives in `content/{en,ar}/posts/<series>/postNN-slug.md`, configured per language in `hugo.toml` (`contentDir`, separate menus). Series so far: `cuda`, `gstreamer`, `numerical-optimization`, `point-cloud-rendering`. Front matter uses `series: ["<name>"]`, `tags`, and `weight: N+1` (Hugo treats weight 0 as unset, so post00 has weight 1). Arabic posts mirror the English ones; Arabic text is wrapped with the `{{< en >}}` shortcode (`<bdi>`) for embedded English terms.

**Code is never pasted into posts.** Source for every post lives in `assets/code/<series>/postNN/` and is embedded with the `{{< code file="cuda/post10/scan.cu" region="name" >}}` shortcode (`layouts/shortcodes/code.html`). Regions are delimited in the source by `// #region name` / `// #endregion` comments, stripped on render. **A missing file or unknown region fails the build** — so renaming a file or region in `assets/code` requires updating every post (EN and AR) that references it. Blocks of 4+ lines render collapsed (`<details>`), via both the shortcode and `layouts/_default/_markup/render-codeblock.html`. "View on GitHub" links use `params.codeRepo`/`codeBranch`, so they only resolve once pushed to `main`.

**Math**: write LaTeX with `\( \)`, `\[ \]`, or `$$`; the passthrough render hook (`render-passthrough.html`) renders it with KaTeX at build time. Set `math: true` in front matter to load the KaTeX CSS (`partials/extra-head.html`). `{{< sidenote >}}` gives margin notes for math newcomers (CSS in `static/css/sidenote.css`, loaded only on pages using it).

**Quiz**: end a post with `{{< quiz >}}` holding YAML questions (`q`, `options`, `answer` index from 0, optional `explain`; usage in `layouts/shortcodes/quiz.html`). A bad `answer` index fails the build. `static/js/quiz.js` sends each first answer to Google Analytics as a `quiz_answer` event.

**i18n**: UI strings in `i18n/{en,ar}.toml`; language flags in `data/langFlags.yaml`. Interactive tools (e.g. `static/tools/frame-types-explorer`) are plain static files.

## Conventions

- Series planning notes live in `.claude/plans/` (e.g. `cuda.md` has the post list, status, and open TODOs). Check them before adding posts to a series.
- Post template (CUDA series): "What We Will Learn", full program, build & run, exercise. Measured numbers are reported against the RTX 4090 reference machine.
- Commits: no `Co-Authored-By` trailer for this user, and commit with the user's own identity. The point-cloud and numerical-optimization posts were fact-checked against external source repos; don't change technical claims casually.
