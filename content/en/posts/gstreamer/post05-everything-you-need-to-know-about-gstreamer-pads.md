---
title: "Post05: Everything You Need to Know About Gstreamer Pads"
date: 2026-07-20T19:15:45+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
---

## Learning objectives

By the end of this post, you should be able to:

- Explain what a pad is, and why elements never connect to each other directly.
- Distinguish source pads from sink pads.
- Read a pad template with `gst-inspect-1.0` and know what it tells you before a pipeline ever runs.
- Explain the three pad availability types — Always, Sometimes, and Request — and when each applies.
- Connect to dynamic pads at runtime using the `pad-added` signal.
- Request pads explicitly, the way `tee` and `nvstreammux` require.
- Explain what ghost pads are and why bins need them.
- Understand pad capabilities (caps) and how negotiation between two pads actually proceeds.
- Debug the most common "could not link" failures.
- Know how DeepStream's own plugins use each pad type — including one case that breaks the pattern you'd expect.

## Why pads matter

In the previous posts, a line like this just worked:

```
filesrc ! qtdemux ! h264parse ! avdec_h264 ! autovideosink
```

But that syntax hides a real question: why do some elements link to each other instantly, while others fail outright with `could not link element A to element B`? And why do elements like `decodebin`, `uridecodebin`, `qtdemux`, and `tee` need extra code — signal handlers, explicit pad requests — that `videotestsrc ! autovideosink` never does?

The answer is pads, and the model is closer to hardware than software. Think of an element the way you'd think of a piece of equipment, and its pads as the physical ports on it. A keyboard and a computer don't need to know anything about each other's internals to work together — they just both need a compatible USB port. GStreamer elements work the same way: `videotestsrc ! autovideosink` isn't really linking two elements directly. Underneath, GStreamer links `videotestsrc`'s source pad to `autovideosink`'s sink pad, and neither element ever refers to the other.

A pad is the actual point where data enters or leaves an element — every buffer that flows through a pipeline crosses one on the way in and one on the way out, with no path that bypasses them. Pads that produce data are **source pads**, conventionally drawn on an element's right, since data flows left to right; pads that consume data are **sink pads**, drawn on the left. `filesrc`, `videotestsrc`, and `nvurisrcbin` are sources with only source pads; `autovideosink`, `filesink`, and `nveglglessink` are sinks with only sink pads; everything in between — `videoconvert`, `h264parse`, `nvinfer` — has at least one of each.

Chain several elements with `!`, and each one is doing the same thing under the hood: walking from one element's source pad to the next element's sink pad. Once that's clear, most of what looks like inconsistent behavior across `decodebin`, `tee`, `nvstreammux`, and friends turns out to follow one small, consistent rule — it's not about the element, it's about what kind of pads it has.

## Pad templates, and how to read them

Before any pad exists, GStreamer knows about it as a **pad template** — a declaration of what a pad *could* be if the element ever creates one. `gst-inspect-1.0` is the fastest way to read those templates, and it's worth running on every unfamiliar element before you try to link it to anything:

```bash
gst-inspect-1.0 vp8dec
```

The relevant section of the output looks like this:

```text
Pad Templates:
  SINK template: 'sink'
    Availability: Always
    Capabilities:
      video/x-vp8

  SRC template: 'src'
    Availability: Always
    Capabilities:
      video/x-raw
                 format: I420
                  width: [ 1, 2147483647 ]
                 height: [ 1, 2147483647 ]
              framerate: [ 0/1, 2147483647/1 ]
```

Three things matter in that output: the pad's name (`sink`, `src`), its **availability**, and its **capabilities** — the media types it can accept or produce. Capabilities alone let GStreamer catch an impossible link before anything runs: if two elements' pad templates share no common capabilities, GStreamer already knows they can never connect, without instantiating either pad. Availability is what decides whether a pad exists the moment the element is created, or only shows up later, or has to be explicitly requested — and that's the concept the rest of this post is really about.

## The three pad availability types

Every pad on every GStreamer element falls into exactly one of three categories.

### Always pads

An Always pad exists the instant the element is created — no signal handlers, no requests, nothing to wait for. This covers most elements you'll use day to day: `videoconvert`, `videoscale`, `queue`, `identity`, and most filters and converters have simple Always pads on both sides.

### Sometimes pads

A Sometimes pad doesn't exist yet when the element is created, because the element genuinely can't know in advance what pads it will need. The canonical example is a demuxer like `qtdemux`: run `gst-inspect-1.0 qtdemux` before it has parsed anything, and you'll see only a `sink` pad. The source pads — `video_0`, `audio_0`, and so on — don't exist until `qtdemux` has actually read the container and discovered what streams are inside it. One MP4 might have video and audio; another might add a subtitle track; a third might have two audio tracks in different languages. There's no way to know the pad layout ahead of time, so GStreamer creates these pads dynamically and fires a `pad-added` signal each time one appears. This is why `qtdemux`, `matroskademux`, `decodebin`, and `uridecodebin` all require a signal handler if you're linking downstream elements manually:

```cpp
g_signal_connect(demux, "pad-added", G_CALLBACK(on_pad_added), decoder);
```

Inside that callback, the usual flow is to inspect the new pad, find a compatible sink pad on the element you want to link it to, and connect them — often adding and syncing new elements into the pipeline from inside the callback itself, since you don't know what you're linking until the pad shows up.

### Request pads

A Request pad is the opposite case: the element *could* create more pads, but only does so when the application explicitly asks. `tee` is the standard example. A freshly created `tee` has no source pads at all; every output branch has to be requested:

```cpp
static void
some_function (GstElement * tee)
{
  GstPad *pad = gst_element_request_pad_simple (tee, "src_%u");
  /* link the pad here */
  gst_object_unref (GST_OBJECT (pad));
}
```

Multiplexers, aggregators, and stream-batching elements work the same way — `multiqueue`, `input-selector`, `output-selector`, `mpegtsmux`, and DeepStream's `nvstreammux` all expose at least one side of their pads as Request rather than Always or Sometimes.

### At a glance

| Type | Pad exists | Typical examples |
|---|---|---|
| Always | Immediately, when the element is created | `videoconvert`, `queue` |
| Sometimes | At runtime, once the element has enough information | `qtdemux`, `decodebin` |
| Request | Only when the application explicitly asks for it | `tee`, `nvstreammux` |

The distinction sounds academic until you're staring at a `pad-added` handler that never fires, or a `NULL` pointer from a pad you assumed already existed. At that point, knowing which of the three you're dealing with is usually the entire fix.

## Ghost pads

A `GstBin` — the container that holds a group of elements — has no pads of its own by default, even though the elements inside it obviously do. **Ghost pads** solve this: a ghost pad exposes an internal element's pad on the bin itself, the way a symbolic link exposes a file somewhere other than where it actually lives. Once a bin has a ghost pad, you can link to the bin directly, without knowing or caring what's inside it — which is exactly what lets you package a reusable chunk of pipeline, like a "smart bin" for license-plate recognition, and treat it as if it were a single element.

## Pad capabilities and negotiation

A pad's capabilities (its **caps**) describe the media types it can accept or produce — things like `video/x-raw, format=NV12, width=1920, height=1080`, or simply `video/x-h264`. Caps live on both pad templates (the full range of what's theoretically possible) and on actual pads (either the still-open set of options, or, once negotiation finishes, the one concrete format that pad is currently using).

Negotiation itself follows a simple, fixed rule: **downstream suggests, upstream decides.** A downstream pad can query its peer for the caps it supports; the upstream pad picks one concrete format from whatever both sides agree on and announces it by sending a `CAPS` event downstream. If a downstream element later needs a different format — say, a `videoscale` whose output size property just changed — it doesn't renegotiate directly. It sends a `RECONFIGURE` event *upstream*, and the upstream element responds by picking a new format and sending a fresh `CAPS` event back down.

This is also why `videotestsrc ! audioconvert` fails immediately: `videotestsrc`'s source pad only ever offers `video/x-raw`-family caps, `audioconvert`'s sink pad only ever accepts `audio/x-raw`-family caps, and the two sets don't intersect. No negotiation is even attempted, because the pad templates alone already rule it out. A more common real-world case is a missing `videoconvert` between two elements that both handle raw video but in incompatible pixel formats — RGB versus NV12, say. The fix there is almost always to insert a converter, not to fight the caps directly.

## Debugging pad and caps issues

Three tools cover almost every "could not link" problem:

- **`gst-inspect-1.0 <element>`** — shows every pad template, its availability, and its capabilities. This is where you start.
- **`gst-launch-1.0 ... -v`** — reruns your pipeline with verbose output, including the caps negotiated on every link, so you can see exactly where two pads landed on an incompatible format.
- **`GST_DEBUG=GST_CAPS:6 gst-launch-1.0 ...`** — turns on GStreamer's internal caps-negotiation logging for a much more detailed trace than `-v` alone provides. Run `gst-launch-1.0 --gst-debug-help` for the full list of categories on your installed version, since the exact set varies slightly by release.

## How DeepStream uses each pad type

Pads matter even more once you're building DeepStream pipelines, and this is where the three categories show up in ways that aren't always obvious from the element names alone.

- **`nvstreammux`** batches frames from multiple sources into a single buffer, and every input is a Request pad — you call `gst_element_request_pad_simple(mux, "sink_%u")` for each source you connect, the same way you'd request a pad from `tee`.
- **`nvstreamdemux`** does the reverse: it splits a batched buffer back into one buffer per source. Despite the name, and despite sitting right next to genuine dynamic demuxers like `qtdemux` in a typical pipeline, `nvstreamdemux`'s output pads are also **Request**, not Sometimes — you have to request `src_%u` explicitly for each stream you want out of the batch, exactly as you would with `tee`. That's a deliberate difference from `qtdemux`: a container demuxer discovers its streams by parsing a bitstream it hasn't seen before, but `nvstreamdemux` doesn't need to discover anything — the batch composition was already decided by the application when it built the `nvstreammux` upstream, so there's nothing left to detect at runtime.
- **`tee`**, used the same way it is anywhere else in GStreamer, duplicates a batch's inference output across multiple downstream branches — one for on-screen display, one for recording, one for a message broker, and so on.

## Common mistakes

- Treating pads and elements as interchangeable, and trying to link elements when you actually need to link their pads.
- Forgetting to connect a `pad-added` handler before setting the pipeline to `PLAYING`, so Sometimes pads appear with nothing listening for them.
- Assuming every element has exactly one source and one sink pad — plenty of real elements have neither, or several of each.
- Assuming a plugin's output pads are Sometimes just because it has "demux" in the name — `nvstreamdemux` is the clearest counterexample.
- Forgetting to release Request pads (`gst_element_release_request_pad`) once you're done with them.
- Chasing a caps negotiation failure by trial and error instead of just reading the pad templates with `gst-inspect-1.0` first.

## Summary

A pad is the actual connection point between elements — the thing that links, not the element itself. Every pad has a direction (source or sink) and an availability (Always, Sometimes, or Request), and that availability alone explains most of the API differences you'll run into across `decodebin`, `tee`, `qtdemux`, `nvstreammux`, and `nvstreamdemux`. Layered on top of that, caps and caps negotiation determine whether two compatible-looking pads can actually agree on a format, and pad templates let GStreamer catch a doomed link before the pipeline ever starts. Once this framework is solid, the plugins that felt like special cases — `decodebin`'s dynamic pads, `tee`'s request pads, `nvstreammux`'s batching, `nvstreamdemux`'s output — turn out to all be instances of the same three rules.

## Exercises

1. Run `gst-inspect-1.0` on `videotestsrc`, `videoconvert`, `decodebin`, `tee`, `qtdemux`, and `nvstreammux`. For each, identify every pad template and its availability.
2. Build a pipeline that uses `tee` to display the same video feed in two separate windows.
3. Build a pipeline with `qtdemux` on a real media file and log every `pad-added` signal it fires.
4. Run any working pipeline with `gst-launch-1.0 -v` and find the line where the final, negotiated caps are printed.
5. Without running it, explain why `videotestsrc ! audioconvert` can never link — and what pad-level information tells you that before you even try.
