---
title: "gstreamer.hpp — Modern C++ for GStreamer Without the Boilerplate"
date: 2026-07-23T10:00:00+03:00
toc: false
images:
tags:
  - gstreamer
  - linux
  - cpp
---

## Learning objectives

By the end of this post, you should be able to:

- Understand why gstreamer.hpp exists and what problem it solves over the raw C API.
- Distinguish the two layers: non-owning `gst::` handles and owning `gst::raii::` types.
- Use `nonstd::expected` to handle errors without checking raw pointers or catching exceptions.
- Build a pipeline using the declarative `gst::PipelineDesc` DSL.
- Add gstreamer.hpp to your own CMake project.

## Where it came from

This library started as something else entirely.

I was building NVIDIA DeepStream inference pipelines at work — the kind that ingest camera streams, run object-detection models on the GPU, and push metadata downstream for further processing. DeepStream is built on top of GStreamer, so every pipeline I wrote meant wrestling with both APIs at once: GStreamer's raw C API for the transport layer, and DeepStream's own plugin API for the AI side.

The C API is workable, but it ages poorly at scale. Every function returns a raw pointer or a `gboolean`. Errors come back as `GError**` out-parameters. Every resource you create has a different unref call, and forgetting one is a silent leak. Nothing is type-safe: element properties are set by name as strings, and a typo at runtime gives you silence instead of a compile error.

So I wrote **deepstream.hpp** — a header-only C++ wrapper that covered both GStreamer primitives and the DeepStream plugin layer under a single include.

About halfway through, I noticed something: the GStreamer half of the library had no DeepStream dependency at all. The `gst::` namespace — handles, RAII wrappers, error types, the pipeline DSL — would compile and work on any machine with GStreamer installed, no NVIDIA GPU required.

That observation drove the split. The GStreamer layer became **gstreamer.hpp**, a standalone repository. deepstream.hpp now vendors it as a submodule and builds the `ds::` layer on top. Anyone who only needs GStreamer gets a library that doesn't pull in NVIDIA-specific headers or require the DeepStream runtime to exist.

## What the raw API looks like

Before showing what gstreamer.hpp offers, it helps to see the problem it is solving. Here is a minimal GStreamer application in straight C:

```cpp
#include <gst/gst.h>

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch("videotestsrc ! autovideosink", &error);
  if (!pipeline) {
    g_printerr("Error: %s\n", error->message);
    g_error_free(error);
    return EXIT_FAILURE;
  }

  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* msg = gst_bus_timed_pop_filtered(
      bus, GST_CLOCK_TIME_NONE,
      (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  if (msg) gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  return EXIT_SUCCESS;
}
```

Every exit path requires the same manual cleanup. Miss an `unref` on the error path and you have a leak. The `GstMessageType` cast is required because the enum values don't define `operator|`. There is no way to know at compile time whether `gst_parse_launch` succeeded without checking the pointer.

## The two-layer design

gstreamer.hpp mirrors the two-layer model used by `vulkan.hpp`:

```
gst::          — non-owning typed handles, free functions, enums, DSL descriptors
gst::raii::    — owning move-only wrappers, implicitly convertible to the layer above
```

The `gst::` layer is thin. `gst::Element` holds a `GstElement*` but does not own it — it is trivially copyable and safe to pass around. Free functions take these handles and return `nonstd::expected<T, E>` instead of raw pointers.

The `gst::raii::` layer wraps the same types with ownership. `gst::raii::Pipeline` holds the element, calls `gst_object_unref` in its destructor, and is move-only. Crucially, it implicitly converts to `gst::Pipeline`, so every free function that accepts a `gst::Pipeline` also accepts a `gst::raii::Pipeline` — you never have to `.get()` to call into the lower layer.

## Error handling with expected

The same example, written with gstreamer.hpp:

```cpp
#include <gstreamer.hpp>

int main(int argc, char* argv[]) {
  gst::init(std::span(argv, static_cast<size_t>(argc)));

  auto pipeline = gst::parse_launch("videotestsrc ! autovideosink");
  if (!pipeline) {
    fmt::print(stderr, "Failed to parse pipeline: {}\n", pipeline.error()->message);
    return EXIT_FAILURE;
  }

  if (auto state = gst::element_set_state(*pipeline, GST_STATE_PLAYING); !state) {
    fmt::print(stderr, "Failed to play: {}\n", state.error());
    return EXIT_FAILURE;
  }

  auto bus = gst::element_get_bus(*pipeline);
  if (!bus) return EXIT_FAILURE;

  auto msg = gst::bus_timed_pop_filtered(
      *bus, GST_CLOCK_TIME_NONE,
      gst::MessageType::Error | gst::MessageType::EOS);

  gst::element_set_state(*pipeline, GST_STATE_NULL);
  return EXIT_SUCCESS;
}
```

A few things changed:

- `gst::parse_launch` returns `nonstd::expected<gst::Element, gst::ErrorPtr>`. The error value carries the `GError` wrapped in a `unique_ptr` that calls `g_error_free` on destruction — no manual free.
- `gst::MessageType` is a strongly-typed flags enum. The `|` operator is defined, so `gst::MessageType::Error | gst::MessageType::EOS` compiles without a cast.
- `gst::element_set_state` returns `nonstd::expected<void, std::string>`, so failed state changes are explicit.
- There are no manual `gst_object_unref` calls because `parse_launch` returned a non-owning handle here. The owning version is one include away.

## Owning resources with the RAII layer

When you want automatic cleanup, switch to `gstreamer_raii.hpp`:

```cpp
#include <gstreamer_raii.hpp>

int main(int argc, char* argv[]) {
  gst::init(std::span(argv, static_cast<size_t>(argc)));

  auto pipeline = gst::raii::parse_launch(
      "videotestsrc pattern=snow ! video/x-raw,width=1280,height=720 ! autovideosink");
  if (!pipeline) {
    fmt::print(stderr, "Failed to create pipeline.\n");
    return EXIT_FAILURE;
  }

  gst::element_set_state(*pipeline, GST_STATE_PLAYING);
  g_main_loop_run(g_main_loop_new(nullptr, FALSE));

  // destructor calls gst_object_unref — nothing to clean up
  return EXIT_SUCCESS;
}
```

`gst::raii::parse_launch` returns `nonstd::expected<gst::raii::Pipeline, gst::ErrorPtr>`. The `Pipeline` destructor calls `gst_object_unref`. Every exit path is now safe, including exceptions, because C++ guarantees destructors run.

## The pipeline DSL

For pipelines built from individual elements rather than a launch string, the library provides a declarative DSL built on `gst::PipelineDesc` and `gst::Node`:

```cpp
auto result = gst::build(gst::PipelineDesc{
    gst::Node{"videotestsrc", "src"}.prop("pattern", 0),
    gst::Node{"videoconvert"},
    gst::Node{"autovideosink"},
});

if (!result) {
    fmt::print(stderr, "Build failed: {}\n", result.error());
    return EXIT_FAILURE;
}

// result.value() is a gst::raii::Pipeline
gst::element_set_state(*result, GST_STATE_PLAYING);
```

`gst::build()` creates every element via `gst_element_factory_make`, sets properties, adds them all to a new pipeline, and links them left to right. If any step fails — unknown factory name, incompatible caps, bad property type — it returns an error string immediately. On success you get an owned `gst::raii::Pipeline` back.

The `.prop()` call accepts a `gst::PropertyValue`, which is a `std::variant` of the types GStreamer properties actually use: `bool`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `double`, and `std::string`. A property set with the wrong type fails at compile time, not at runtime when GStreamer silently ignores the value.

## Tutorials

The repository ships 25 working tutorials across three difficulty tiers. Each tutorial builds the same pipeline three ways: raw C, `gst::raii` wrappers, and the `gst::PipelineDesc` DSL — so you can compare all three approaches on identical problems.

```
tutorials/
├── easy/       HelloWorld, VideoFilePlayer, WebcamViewer, AudioPlayer,
│               CapsAndFilters, ElementByHand, StatesAndSeeking, PipelineBuilder
├── medium/     DynamicPipeline, EventsAndQueries, BuffersAndMemory, ClocksAndSync,
│               CPUVideoProcessing, ImageCapture, PipelineInspector,
│               RTSPClient, TagsAndMetadata, VideoRecorder
└── hard/       CustomPlugin, CustomSourceSink, EncodeProfiles,
                MultiCameraViewer, NetClockSync, RTSPServer, TestingElements
```

The hard tier includes tutorials that go below the gstreamer.hpp API — writing a `GstBaseTransform` subclass, unit-testing a custom element with `GstHarness`, and synchronising pipelines across machines with `GstNetClock`. Those tutorials drop to the raw C API where the wrapper has no coverage, and the READMEs name the gap explicitly.

## Getting started

The library is header-only. Clone the repo, then point CMake at it:

```cmake
add_subdirectory(third_party/gstreamer.hpp)
target_link_libraries(my_target PRIVATE gstreamer::hpp)
```

Build requirements are CMake 3.20+, a C++20 compiler, and GStreamer with the Video component. `expected-lite` and `fmt` are vendored as submodules and require no separate installation.

```bash
git clone --recurse-submodules https://github.com/wow2006/gstreamer.hpp
cmake -B build -S .
cmake --build build
ctest --test-dir build
```

146 tests pass on a clean build.

## Summary

gstreamer.hpp started as the GStreamer half of a larger DeepStream wrapper. When it became clear that half had no DeepStream dependency at all, I extracted it into a standalone library so it could be useful to anyone writing GStreamer pipelines in C++, not just those running NVIDIA inference workloads.

The library gives you RAII ownership with `gst::raii::*`, `nonstd::expected`-based error handling on every function that can fail, strongly-typed enums and flags where GStreamer uses integers and casts, and a declarative pipeline DSL that replaces the create-add-link loop with a single `gst::build()` call. The two layers are designed to be mixed freely — owning and non-owning types convert to each other implicitly, so you can adopt the wrapper incrementally rather than rewriting a codebase all at once.

## Exercises

1. Clone the repository and run `ctest`. Verify all 146 tests pass on your machine.
2. Build the `HelloWorld` tutorial and compare `main.cpp`, `main_raii.cpp`, and `main_view.cpp` side by side. Count the lines of cleanup code in each.
3. Write a pipeline using `gst::PipelineDesc` that captures from a webcam and writes to a file. What happens when you provide a factory name that does not exist?
4. Add a `.prop()` call to a `gst::Node` and pass a value of the wrong type. What error do you get, and at what stage?
5. Pick any medium tutorial and trace how the `gst::raii::` type converts to the `gst::` handle when passed to a free function.
