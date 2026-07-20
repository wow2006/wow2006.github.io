---
title: "Post04: Playing Video File With Gstreamer"
date: 2026-06-19T22:09:06+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
---

In the previous post, we created an H.264 video stored inside an MKV container using `videotestsrc`. While this is useful for learning GStreamer fundamentals, real applications typically process existing media files.

In this post, we will learn how to load and play a video file using GStreamer.

## Contents

* Download a sample video
* Inspect the media file
* Play the video using GStreamer
* Understand `filesrc`
* Understand `decodebin`
* Understand `videoconvert`
* Learn about dynamic pads

---

## Download a Sample Video

We will use the well-known **Big Buck Bunny** sample video.

You can download it using your web browser or from the command line:

> IMPORTANT: The video file size is `398M`. You are free to download any H264 video.

```bash
curl -o big_buck_bunny_720p_h264.mov \
https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov
```

---

## Inspect the Media File

Before building a pipeline, it is useful to understand the contents of the media file.

Open the file using VLC and view:

```text
Tools → Codec Information
```

or press:

```text
Ctrl + J
```

You should see information similar to:

```text
Stream 0: (Video)
    Codec: H264 - MPEG-4 AVC (part 10) (avc1)
    Resolution: 1280x720
    Frame Rate: 24 FPS

Stream 1: (Audio)
    Codec: MPEG AAC Audio (mp4a)
    Bitrate: 448 kb/s
    Sample Rate: 48000 Hz
    Bits Per Sample: 16
```

The file contains two streams:

1. H.264 video
2. AAC audio

> A MOV file is a multimedia container format developed by Apple as part of the QuickTime framework. Like MP4 and MKV, it can contain multiple streams such as video, audio, subtitles, and metadata. The container format is separate from the codecs stored inside it.

---

## Playing the Video

Let's start with a simple GStreamer pipeline:

```bash
gst-launch-1.0 \
filesrc location=big_buck_bunny_720p_h264.mov \
! decodebin \
! videoconvert \
! autovideosink
```

Running this command opens a video window and starts playback.

Although the source file contains an audio stream, this pipeline only displays the video stream.

---

## Understanding the Pipeline

### `filesrc`

`filesrc` reads raw bytes from a file on disk.

```text
File
  ↓
filesrc
```

At this stage, GStreamer does not know whether the file contains video, audio, subtitles, or any other type of data.

---

### `decodebin`

`decodebin` is one of the most useful elements in GStreamer.

It automatically:

1. Detects the container format.
2. Creates the appropriate demuxer.
3. Creates the required decoders.
4. Exposes decoded streams.

For our MOV file, `decodebin` internally creates elements similar to:

```text
MOV File
    ↓
qtdemux
 ├─ H264 Stream
 └─ AAC Stream

H264 Stream
    ↓
H264 Decoder
    ↓
Raw Video
```

As a result, the output from `decodebin` is no longer compressed H.264 data but raw video frames.

---

### `videoconvert`

Video sinks often require specific pixel formats.

`videoconvert` automatically converts between formats when necessary.

```text
Raw Video
    ↓
videoconvert
    ↓
Compatible Raw Video
```

Using `videoconvert` makes pipelines more portable across platforms and graphics backends.

---

### `autovideosink`

`autovideosink` automatically selects an appropriate video output element for your system.

Depending on the operating system, it may choose different sinks internally.

```text
Raw Video
    ↓
autovideosink
    ↓
Screen
```

---

## Dynamic Pads

Unlike `filesrc` or `videoconvert`, `decodebin` does not know in advance how many streams a file contains.

A file may contain:

* Video only
* Audio only
* Video and audio
* Multiple audio tracks
* Subtitles

For this reason, `decodebin` creates its output pads dynamically after analyzing the file.

```text
decodebin
 ├─ video pad
 └─ audio pad
```

When using `gst-launch`, GStreamer automatically handles these dynamic pads.

However, when writing C++ or Rust applications, we must listen for the `pad-added` signal and connect the pads ourselves.

This will be an important topic in upcoming posts.

---

## Summary

In this post, we learned how to play an existing video file using GStreamer.

We introduced:

* `filesrc` for reading files.
* `decodebin` for automatic demuxing and decoding.
* `videoconvert` for pixel format conversion.
* `autovideosink` for displaying video.
* Dynamic pads created by `decodebin`.

In the next post, we will use `gst-discoverer-1.0` to inspect media files and explore their streams directly from the command line.

---

## Exercises

1. Replace `autovideosink` with a platform-specific video sink.
2. Run the pipeline on different media files and observe the behavior of `decodebin`.
3. Use `gst-inspect-1.0 decodebin` to learn more about its properties.
4. Try playing a file that contains only audio and observe what happens.

---

## Questions

1. What is the difference between a container and a codec?
2. Why do we need `decodebin` in this pipeline?
3. What problem does `videoconvert` solve?
4. Why does `decodebin` create pads dynamically?
5. Why is the AAC audio stream ignored in our pipeline?
