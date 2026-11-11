# Salt and Pepper

Source of [wow2006.github.io](https://wow2006.github.io/), a bilingual (English / Arabic) blog about computer graphics, vision, and GPU programming. Built with [Hugo](https://gohugo.io/) and a fork of the [hello-friend-ng](https://github.com/wow2006/hugo-theme-hello-friend-ng) theme.

## Series

- **CUDA** — zero to hero, with measured kernels
- **GStreamer** — C++ and Rust pipelines
- **Numerical optimization**
- **Point-cloud rendering**

## Build

The theme is a git submodule:

```bash
git clone --recurse-submodules https://github.com/wow2006/wow2006.github.io.git
cd wow2006.github.io
hugo server --buildDrafts   # preview at http://localhost:1313
hugo --buildDrafts          # full build into public/
```

## Layout

| Path | Contents |
|------|----------|
| `content/{en,ar}/posts/<series>/` | Posts, one folder per series |
| `assets/code/<series>/postNN/` | Source for each post, embedded with the `code` shortcode |
| `layouts/` | Shortcodes and render hooks (code blocks, KaTeX math, sidenotes) |
| `i18n/`, `data/` | UI strings and language flags |
| `static/` | CSS, images, standalone interactive tools |

The code samples are standalone CMake projects:

```bash
cd assets/code/cuda/post10
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

See [CLAUDE.md](CLAUDE.md) for authoring conventions.
