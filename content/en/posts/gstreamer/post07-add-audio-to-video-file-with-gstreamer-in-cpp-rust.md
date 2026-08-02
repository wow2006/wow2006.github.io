---
title: "Post07: Adding Audio to a Video File with GStreamer in C++20 and Rust"
date: 2026-07-28T22:15:03+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
  - cpp
  - rust
---

Post06 played `big_buck_bunny_720p_h264.mov` silently. The file has an AAC audio stream in it, `decodebin` decoded that stream, and our `pad-added` callback threw it away:

```cpp
if (!pad_type.starts_with("video/x-raw") || gst_pad_is_linked(convert_sink)) {
    return;
}
```

That early return was the whole reason there was no sound. Nothing else in the pipeline was missing audio support — there simply was no audio branch for the pad to link to. This post adds one: a second sink chain, and a callback that picks a destination based on what the pad is carrying instead of rejecting everything that isn't video.

This is Exercise 1 from Post06, worked through in both languages.

## Prerequisites

Post06's setup carries over unchanged — same toolchain, same `big_buck_bunny_720p_h264.mov`. The Rust side needs no new crates; the C++ side needs no new `pkg-config` module. `audioconvert` and `autoaudiosink` live in `gst-plugins-base` and `gst-plugins-good`, which you already installed back in Post01.

## Project Source Code

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post07/cpp/main.cpp" region="create-project" link="false" >}}

{{< code file="gstreamer/post07/cpp/main.cpp" region="code" link="false" >}}

Compile and run it:

{{< code file="gstreamer/post07/cpp/main.cpp" region="build-run" link="false" >}}
{{< /tab >}}

{{< tab "Rust" >}}
{{< code file="gstreamer/post07/rust/src/main.rs" region="create-project" link="false" >}}

{{< code file="gstreamer/post07/rust/src/main.rs" region="code" link="false" >}}

Run it:

{{< code file="gstreamer/post07/rust/src/main.rs" region="build-run" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

## Creating the Elements

Six elements now instead of four. `filesrc` and `decodebin` are unchanged and still shared by both streams — `decodebin` demuxes the container and decodes *both* elementary streams whether we consume them or not. What's new is that the single `videoconvert → autovideosink` tail becomes two independent tails:

- `videoconvert → autovideosink`
- `audioconvert → autoaudiosink`

Note the names in the C++ version — `video_convert` and `audio_convert` are not cosmetic. The callback looks elements up by name at runtime, which we'll get to below.

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post07/cpp/main.cpp" region="code-create-elements" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post07/rust/src/main.rs" region="code-create-elements" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

`audioconvert` is doing the same job on the audio side that `videoconvert` does on the video side: negotiating a format the sink can actually take. `decodebin` hands us whatever the AAC decoder produces — some particular sample format, channel layout and interleaving — and the output device wants something specific of its own. `audioconvert` sits between them and reconciles the two.

## Linking What We Can, Upfront

Three static links this time instead of two, one per Always-pad pair:

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post07/cpp/main.cpp" region="code-link-elements" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post07/rust/src/main.rs" region="code-link-elements" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

Before the pipeline starts, we have `filesrc → decodebin` on one side and two disconnected tails on the other. Both gaps close from the same callback.

## Connecting Decodebin's Dynamic Pads

Post06's callback answered a yes/no question: is this pad video? This one answers a routing question: *which* branch does this pad belong to? Same signal, same two firings, but now both firings do work.

The signal connection is where the two languages diverge, so it's worth looking at directly:

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post07/cpp/main.cpp" region="code-decoder-signal" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post07/rust/src/main.rs" region="code-decoder-signal" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

C++ hands the callback the pipeline and lets it find the right converter by name; Rust captures both sink pads in the closure and picks between them. Both are fine — see [Two Ways to Reach the Right Converter](#two-ways-to-reach-the-right-converter) for why they ended up different.

The callback itself:

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post07/cpp/main.cpp" region="code-pad-added" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post07/rust/src/main.rs" region="code-pad-added" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

The shape is the same in both: read the caps, take the media type from the first structure, map it to a sink pad, bail out on anything we don't recognise, then link. The `else { return; }` branch matters — `decodebin` will happily hand you a subtitle or timed-text pad from a container that has one, and linking that into `audioconvert` fails at caps negotiation rather than at `gst_pad_link`, which is a much less obvious error to read.

The already-linked check that Post06 called a subtlety is load-bearing now. A file with two audio tracks fires `pad-added` twice with `audio/x-raw`, and `audioconvert`'s sink pad is an Always pad — exactly one of them can be linked. Without the guard, the second link attempt returns `GST_PAD_LINK_WAS_LINKED` and prints an error for a file that's otherwise perfectly playable. With it, the first track wins and the rest are ignored.

## Two Ways to Reach the Right Converter

C++ passes `pipeline` as the callback's `user_data` and calls `gst_bin_get_by_name` to fetch the converter:

```cpp
const ElementPtr converter{
    gst_bin_get_by_name(GST_BIN(pipeline), converter_name)};
```

`gst_bin_get_by_name` returns a *new reference*, which is why the result goes straight into an `ElementPtr` — the lookup is a refcount increment, and forgetting the matching unref here leaks one element per pad, in a callback, which is the worst possible place to put a leak. `ElementPtr` makes it a non-decision.

Rust captures the two pads directly instead:

```rust
decoder.connect_pad_added(move |_decoder, src_pad| {
    on_pad_added(src_pad, &video_sink_pad, &audio_sink_pad);
});
```

This is the same constraint from Post06, unchanged: the closure is owned by `decodebin`, which is owned by `pipeline`, so a closure capturing `pipeline` would close a reference cycle GObject's refcounting can't see through. Capturing the two sink pads sidesteps it, and it's cheaper anyway — no by-name lookup on the streaming thread. C++ could capture the pads the same way with a small struct in `user_data`; the by-name lookup is shown here because it's the pattern you'll meet constantly in existing GStreamer C code, and because it scales to callbacks that need to reach elements they weren't handed upfront.

The C++ version does pay for that flexibility: `gst_bin_get_by_name` walks the bin's children comparing strings, and a typo in `"audio_convert"` is a runtime `nullptr` rather than a compile error. Rust's captured pads are checked when `main` runs, before the pipeline ever moves.

## Running It

```bash
./gstreamer-play-file big_buck_bunny_720p_h264.mov
# Dynamic pad created, type: video/x-raw
# Dynamic pad created, type: audio/x-raw
# End of stream
```

Same two lines as Post06, but this time both pads get linked, and Big Buck Bunny plays with sound.

Audio and video stay in sync without us doing anything about it, which is worth pausing on. Both sinks are clocked against the same pipeline clock and both branches carry the same timestamps `decodebin` put on the buffers, so each sink renders its own buffers when their presentation time arrives. The synchronisation isn't something the application arranges — it falls out of the clock and the timestamps.

## Summary

- The audio was already being decoded in Post06; the callback's `return` on non-video pads was all that stood between us and sound.
- Two sink branches now: `videoconvert → autovideosink` and `audioconvert → autoaudiosink`, both statically linked upfront, both waiting on a Sometimes pad from `decodebin`.
- The `pad-added` callback routes on the media type from the pad's caps instead of filtering for one, and returns early on anything that's neither `video/x-raw` nor `audio/x-raw`.
- The already-linked guard stops a second audio track from producing a spurious `GST_PAD_LINK_WAS_LINKED` error.
- C++ reaches the converter with `gst_bin_get_by_name` (a new reference — hold it in an `ElementPtr`); Rust captures both sink pads in the closure, still avoiding a capture of `pipeline` that would form a reference cycle.
- A/V sync comes free from the shared pipeline clock and the decoder's timestamps.

## Exercises

1. Put a `queue` between `decodebin` and each converter. Play a file where the audio and video interleave badly and compare — this is the difference between one thread pushing both branches and two threads doing it independently.
2. Replace `autoaudiosink` with `audioresample ! autoaudiosink` and play a file whose sample rate the output device doesn't support natively. What does the error look like without `audioresample`?
3. Handle `no-more-pads` in addition to `pad-added`, and report which branches ended up linked once `decodebin` says it's done. Run it against an audio-only file.
4. Use `GST_DEBUG_DUMP_DOT_DIR` to dump the pipeline graph after `pad-added` has fired twice, and look at what `decodebin` actually built inside itself for each branch.
