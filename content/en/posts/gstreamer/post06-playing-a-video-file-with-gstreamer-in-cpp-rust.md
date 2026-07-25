---
title: "Post06: Playing a Video File with GStreamer in C++20 and Rust"
date: 2026-07-25T12:28:10+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
  - cpp
  - rust
---

In Post04, we played a video file with a single `gst-launch-1.0` pipeline:

```bash
gst-launch-1.0 filesrc location=big_buck_bunny_720p_h264.mov ! decodebin ! videoconvert ! autovideosink
```

That one-liner works because `gst-launch-1.0` links `decodebin`'s output for us as soon as it appears. Post05 covered exactly what that output is: a Sometimes pad, created at runtime once `decodebin` has identified what's inside the container. Outside of `gst-launch-1.0`, nothing links a Sometimes pad automatically — the application has to do it.

In this post we build the same pipeline directly against the GStreamer API, first in C++20, then in Rust, and connect `decodebin`'s pad ourselves.

Post03 used the plain GStreamer C API from a `.cpp` file, unref'ing everything by hand once we were done with it. That pipeline was fully static, so every `gst_object_unref` had one obvious, unconditional place to live. This pipeline isn't static — a pad turns up mid-flight, an element can fail to create, a link can fail — so this time the C++ side leans on `std::unique_ptr` and a couple of other C++20 additions to make those failure paths harder to get wrong. More on that in [Managing Object Lifetimes](#managing-object-lifetimes).

## Prerequisites

If you worked through Post03, your toolchain is already set up for both languages. This post reuses the same `big_buck_bunny_720p_h264.mov` file we downloaded in Post04 — grab it now if you don't already have it locally.

## Project Source Code

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post06/cpp/main.cpp" region="create-project" link="false" >}}

{{< code file="gstreamer/post06/cpp/main.cpp" region="code" link="false" >}}

Compile and run it:

{{< code file="gstreamer/post06/cpp/main.cpp" region="build-run" link="false" >}}
{{< /tab >}}

{{< tab "Rust" >}}
{{< code file="gstreamer/post06/rust/src/main.rs" region="create-project" link="false" >}}

{{< code file="gstreamer/post06/rust/src/main.rs" region="code" link="false" >}}

No new crates needed beyond what Post03 already added — `pad-added` is part of the base `gstreamer` crate. Run it:

{{< code file="gstreamer/post06/rust/src/main.rs" region="build-run" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

## Creating the Elements

Four elements this time, one more than Post03's chain: `filesrc`, `decodebin`, `videoconvert`, `autovideosink`. `decodebin` is the one doing the interesting work — it reads the container, finds the H.264 stream inside, locates a decoder for it, and exposes the decoded result on a pad it creates once all of that has settled.

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post06/cpp/main.cpp" region="code-create-elements" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post06/rust/src/main.rs" region="code-create-elements" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

## Linking What We Can, Upfront

Only two of the three connections in this pipeline can be made before the pipeline starts running. `filesrc`'s source pad and `decodebin`'s sink pad are both Always pads, so that link is static, and the same is true of `videoconvert` and `autovideosink`. `decodebin`'s *output* is the one Sometimes pad in the chain, so it's the one link we can't make yet:

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post06/cpp/main.cpp" region="code-link-elements" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post06/rust/src/main.rs" region="code-link-elements" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

After this, the pipeline looks like `filesrc → decodebin` on one side and `videoconvert → autovideosink` on the other, with a gap in the middle that only closes once the pipeline starts moving data.

## Connecting Decodebin's Dynamic Pad

`decodebin` fires a `pad-added` signal every time it exposes a new pad — once per elementary stream it finds. `big_buck_bunny_720p_h264.mov` has both an H.264 video stream and an AAC audio stream, so the callback below runs twice, and we only want to act on the video one, exactly like the `gst-launch-1.0` pipeline in Post04 did implicitly.

Before writing the callback, we need `videoconvert`'s sink pad — the thing we're eventually going to link to — and the signal connection itself:



Inside the callback, we read the new pad's negotiated caps to decide whether it's the stream we want, link it if so, and leave it alone if not:

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post06/cpp/main.cpp" region="code-pad-added" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post06/rust/src/main.rs" region="code-pad-added" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

(the full listings above check the link result instead of discarding it — trimmed here for readability). Run either version against the sample file and the callback fires twice:

```
Dynamic pad created, type: video/x-raw
Dynamic pad created, type: audio/x-raw
```

Only the first one gets linked. The second is left dangling on purpose — `decodebin`'s internal `multiqueue` drops data for pads nobody connected rather than blocking the streams that *are* connected, so the audio track is silently discarded instead of stalling playback.

The already-linked check matters for a subtler reason too: `gst_pad_link` on a sink pad that's already linked is an error, and there's nothing stopping a signal from firing again for reasons that have nothing to do with our two streams — a second video track, a redundant emission — so it's worth guarding against even here.

## Managing Object Lifetimes

Post03's pipeline had one shape, decided entirely at startup, so every `gst_object_unref` had one obvious, unconditional place to live. This pipeline creates five reference-counted objects before we know whether all of them will succeed, connects a signal in between, and has three separate points that return early. Matching every one of those with the right manual cleanup call is exactly the kind of bookkeeping that's easy to get right once and wrong the second time someone touches the function.

`std::unique_ptr` with a small custom deleter turns that bookkeeping into something the compiler enforces instead of something we have to remember:

```cpp
struct ElementDeleter {
    void operator()(GstElement* element) const noexcept {
        if (element) gst_object_unref(element);
    }
};
using ElementPtr = std::unique_ptr<GstElement, ElementDeleter>;
```

`gst_element_factory_make` returns `nullptr` on failure — a missing plugin, a typo in the factory name — and every `ElementPtr` above unwraps that into a destructor call automatically on an early return. `gst_bin_add_many` takes ownership of whatever we hand it, so once an element is successfully added, `.release()` hands that ownership over cleanly instead of leaving a dangling owning pointer behind:

```cpp
gst_bin_add_many(GST_BIN(pipeline.get()), source.release(), decoder.release(),
                  convert.release(), sink.release(), nullptr);
```

From that point on, `pipeline`'s own destructor is the only cleanup call left in the program — unreffing the pipeline cascades to every element it owns. `std::string_view::starts_with` and `std::format`, both new in C++20, are the other two additions doing real work here: the first replaces `g_str_has_prefix` with something that reads like a sentence, the second replaces `printf`-style formatting with something checked at compile time.

Rust gets the equivalent safety from ownership and `Drop` directly, without needing a wrapper type — nothing above needed one. The one place Rust asks for more thought is exactly the place C++ doesn't have to: what the `pad-added` closure is allowed to capture. `decodebin` is owned by `pipeline`, and `decodebin` will own this closure once it's connected — so a closure that captures a strong reference to `pipeline` itself would create a cycle the reference counter can't see through, and none of the three objects involved would ever be freed. `convert_sink_pad` doesn't have that problem, since nothing in that cycle holds a reference back to it, which is why the closure captures the pad and not the pipeline.

## Running It

Both versions produce the same result as Post04's `gst-launch-1.0` pipeline: a window opens and Big Buck Bunny plays, silently, since we never link an audio branch.

```bash
./gstreamer-play-file big_buck_bunny_720p_h264.mov
# Dynamic pad created, type: video/x-raw
# Dynamic pad created, type: audio/x-raw
# End of stream
```

## Summary

- We rebuilt Post04's `filesrc ! decodebin ! videoconvert ! autovideosink` pipeline against the GStreamer API, in C++20 and in Rust.
- `filesrc → decodebin` and `videoconvert → autovideosink` are static links made upfront; `decodebin`'s output is a Sometimes pad (Post05), so it's linked from a `pad-added` callback instead.
- The callback checks each new pad's negotiated caps and links only `video/x-raw`, matching what Post04's `gst-launch-1.0` pipeline did automatically. The unlinked audio pad doesn't block anything, since `decodebin`'s internal `multiqueue` drops data for pads nobody connected.
- In C++20, `std::unique_ptr` with small custom deleters replaces manual `gst_object_unref` calls, `.release()` hands ownership to the bin cleanly, and `std::string_view::starts_with` / `std::format` clean up a couple of spots that used to be plain C calls.
- In Rust, `Drop` gives the same safety without a wrapper, but the `pad-added` closure has to avoid capturing `pipeline` itself, to avoid a reference cycle GObject's refcounting can't detect.

## Exercises

1. Extend the pipeline to also play the AAC audio stream: create `audioconvert ! audioresample ! autoaudiosink`, add them to the pipeline, and link the second pad the callback currently ignores.
2. Insert a `queue` element between `decodebin` and `videoconvert`. What changes about which thread the callback runs on?
3. Point the program at a file that contains only an audio stream and see what each language does when the video branch never gets linked.
4. In C++, add a `requires` clause to `ElementDeleter` and its siblings so the deleter template can't be instantiated with a type it doesn't actually know how to unref — a small, safe first use of a C++20 concept.

Coming up: routing that AAC audio stream properly, and — still on the board since Post04 — taking `gst-discoverer-1.0` for a spin before we build a pipeline instead of after.
