---
title: "Post 28: Benchmarking a Renderer Honestly"
date: 2026-10-15T22:00:00+03:00
weight: 29
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - benchmarking
  - testing
---

Every post in this series quotes a number. A number is only worth quoting if you can say how it was taken, reproduce it, and tell when it stops being true. This post collects the habits that came out of the project's own documents (`BENCH.md`, `ROADMAP.md`, `FIXES.md`) and the mistakes recorded in them. It is a methods post: there is little new code, and a lot of discipline.

## The ledger: `BENCH.md`

Every performance claim in the roadmap must **point at a row in `BENCH.md`**. The file's rules:

- **Add rows, do not overwrite.** A later measurement gets its own row with its own commit, so before and after stay visible.
- **Every row names its commit** (or "uncommitted, base X"), the command line, the dataset, the resolution, and the machine.
- **A machine table sits at the top**: GPU and driver, whether clocks were locked, CPU, disk, OS and compiler, display, and the resolution the benches use (2400 x 1350 so the numbers are comparable across sections).
- **A "How to reproduce" block** lists every bench command, so the next person can rerun a row without asking.
- **A different machine gets its own heading**, not a quiet substitution. The mesh-shader rework ([Post 16](../post16-mesh-shaders-for-points/)) was measured on a throttled laptop GPU, and the section says so, with the thermal state.

A new CPU hot path also gets a Google Benchmark `BM_` case (`pce_bench`, built without profiler markers, median of 3 repetitions), and every fix needs a measured before/after (the reason BENCH.md was started at all).

## The mechanics of a GPU frame-time number

These are the rules stated in [Post 01](../post01-why-gl-points-dies/), here as a checklist:

1. **GPU time from `GL_TIME_ELAPSED`**, read from a ring of queries that is never re-begun while unread and never waited on. `GL_TIMESTAMP` marks give the split between raster and resolve inside the scene pass (a `GL_TIME_ELAPSED` query cannot nest inside another).
2. **Vsync off, frame-indexed camera path, warm-up first** (30 to 60 frames), then 300 to 600 frames: report p50, p99 and max, never just the mean.
3. **Fence pacing** at two frames in flight, or the CPU numbers are meaningless.
4. **Clocks logged** (`nvidia-smi -lms 250`) and written into the row. Locking clocks needs admin rights, which the author did not have, so the unlocked state is stated rather than hidden. Early runs saw the same scene go from 4.3 to 7.6 ms when the card dropped a power state.
5. **A pinned view per row**: fit-all, close, fly. The headline "18-20x faster" was true for fit-all only; the same document later states that close-up it is 1.9x and the fly path 7.3x at p50 but 19.5x at p99.
6. **A throttling GPU needs interleaved runs**: old, new, reference, back to back, twice, so thermal drift hits all of them.

Cold-disk numbers use a *fresh unbuffered copy* (`robocopy /J`), since there is no non-admin way to flush the cache on Windows ([Post 20](../post20-streaming-from-a-cold-disk/)).

## Verification is part of the benchmark

A fast wrong answer is a regression. So the same executables that time the renderer also check it:

- `pce --bench` renders each view with each renderer and **diffs the images** against `GL_POINTS`, split into colour and coverage differences, with explicit `FAIL` lines for the paths that must be exact ([Post 11](../post11-image-parity/)).
- `pce --bench-edit` **drives the real UI**: a script that feeds events through Dear ImGui's input queue (`AddKeyEvent`, `AddMousePos`, `AddMouseButtonEvent`), then compares every edit op (13 of them) against a **CPU mirror**: flags and classes byte for byte, undo/redo round trips, the cap path, cloud removal. It fails any frame over 100 ms while a job runs, and checks the Cancel path.
- `pce --bench-lod` checks that resident GPU positions, colours, intensities and classes equal a CPU decode of the file.
- The unit tests (Catch2, one executable, each `TEST_CASE` also its own `ctest` entry, module tags as labels) cover the GL-free halves: chunking, range lookup, renderer choice, undo deltas, cull planes, the LOD cache.

Each fix is one commit and **leaves at least one test that fails if the fix breaks**. That is checked by **mutation**: break the code on purpose and confirm that a test goes red. The notes record the mutants and, importantly, the ones that did *not* get caught and why (dropping the cull rounding margin changes 0 pixels on the bench views, so a separate unit test guards it).

## Things that went wrong, written down

A benchmark log that only records successes is advertising. The documents record:

- **A checker bug**: the background colour `0.1` reads back as `25`, but the check looked for `26`, so "coverage" was always zero. Corrected, with a note and a re-measurement ([Post 11](../post11-image-parity/)).
- **Suspicious equal numbers**: three Morton-bit settings gave identical medians across two runs, and the note says "suspicious — redo with clocks logged". It is still open.
- **A missed target**: first point on `toronto16` took 17.6-18.2 s against a 15 s goal; it was reported as missed, then fixed and re-measured ([Post 22](../post22-previewing-while-converting/)).
- **Focus interference**: the bench window must not take focus. A focused but unhovered window receives the real cursor position from the GLFW backend every frame, after the script's own events, and 7 of 14 mouse checks failed. The fix is `GLFW_FOCUS_ON_SHOW` off and a note that an occasional failure of the mouse checks is "the known focus interference", not the change under test.
- **Pre-existing failures**: where a step fails on the main branch too (the toronto16 pick-and-remove), the row says so, and shows the main-branch run that proves it.
- **An unexplained 35x**: the same GPU kernel is about 35x slower on one Linux driver than on Windows, "not investigated further". It is listed rather than dropped.

## Success criteria before measuring

For the block-culling work ([Post 15](../post15-block-culling-in-the-compute-raster/)) the targets were written first: close view at least 2x faster, fit view no more than 2 % slower, and a **kill condition** (under 1.3x close-up means revert). The result, 3.2x and +1.7 %, is reported as "just inside" the second limit. A "Done when" check closes every task in the roadmap, and a measured shortfall is recorded next to it ("RESEARCH R1.1's 'within 15 s' is missed by ~3 s").

## Shared hardware is a resource

When several agents or people share the GPU, a timing run is ruined by anyone else's job. The project's solution is a **lock directory**: before any `pce --bench*`, `pce_bench` or `--convert` timing, create `.bench-lock/` with `New-Item -ErrorAction Stop` and write an `owner.txt` saying who and what. Creating a directory is atomic, so a second taker fails instead of racing. Builds and `ctest` do not need the lock. The Task 2.3 section also lists what else was running during its measurements (a Python job on 8-20 cores, another ML job on the second GPU), because that load is part of the result.

## What is still missing

The roadmap item 3.5, "frame times at 10 M / 100 M / 500 M points on target GPUs", is not done. The tooling exists and the data sets for 78 M, 118 M and 235 M points exist, but a 10 M and a 500 M row and a second GPU are open. The honest summary of the series' numbers is therefore: **one machine** (an RTX 4090 on Windows plus one Linux and one laptop run, each labelled), at the sizes listed.

## Checklist

1. **One ledger, rows appended**, each with commit, command, dataset, machine, clocks.
2. **Medians and tails** from a warm run on a pinned path; interleave runs on unstable hardware.
3. **Verify while you time**: image diffs with FAIL conditions, CPU mirrors for GPU edits, UI scripts through the real input path.
4. **Mutate your tests** and record what survives.
5. **Write targets and kill conditions before the first run.**
6. **Log your mistakes and your open questions** in the same file as the numbers.
7. **Label every machine**, and refuse to compare across labels without saying so.

Next: the tools for looking inside a frame, once the numbers say something is wrong: [Post 29](../post29-profiling-with-tracy-and-nsight/).

## Where this lives in the code

`docs/BENCH.md`, `docs/ROADMAP.md`, `docs/FIXES.md`, `docs/status.md` ("Timing runs: take the bench lock"), `src/app/bench/` (`bench_frame.cpp`, `bench_path.cpp`, `bench_ui.cpp`, `bench_lod.cpp`, `bench_util.cpp`), `benchmarks/bench.cpp`, `tools/coverage.sh`, `tests/`.
