---
title: "Create a Smaller Video File with H.264 Encoding"
date: 2026-06-11T22:53:50+03:00
draft: true
toc: false
series: ["gstreamer"]
images:
tags:
  - gstreamer
  - linux
---

In the previous post, we created our first video file using GStreamer. The pipeline worked correctly, but there was one major problem: the generated file was very large.

The reason is simple. We stored every video frame as raw, uncompressed image data. While this is easy to understand, it is highly inefficient for storage and distribution.

In this post, we will introduce video compression using the H.264 codec and see how dramatically it reduces file size.

## What You Will Learn

* How to create a video file using H.264 encoding
* The purpose of new GStreamer elements in the pipeline
* The basics of video compression
* Why encoded video files are much smaller than raw video files

---

## Creating an H.264 Video File

The following pipeline generates a test video, compresses it using H.264, and stores it inside a Matroska (`.mkv`) container.

```bash
gst-launch-1.0 -e \
    videotestsrc num-buffers=90 ! \
    video/x-raw,width=1280,height=720,framerate=30/1 ! \
    x264enc ! \
    h264parse ! \
    matroskamux ! \
    filesink location=test.mkv
```

After the pipeline finishes, play the file using VLC, MPV, or any media player that supports H.264 video.

```bash
vlc test.mkv
```

The video should look identical to the one generated in the previous post.

Now let's inspect the file size:

```bash
ls -lh test.mkv
```

Example output:

```bash
-rw-r--r-- 821K test.mkv
```

The file is only **821 KB**.

In the previous post, the same video stored as raw frames was approximately **159 MB**.

| Format        | File Size |
| ------------- | --------- |
| Raw Video     | ~159 MB   |
| H.264 Encoded | ~821 KB   |

This means the encoded file is roughly **193 times smaller** while maintaining visually similar quality.

That is the power of video compression.

---

## Understanding the Pipeline

Let's examine the new elements that were added.

### videotestsrc

```text
videotestsrc
```

Generates synthetic video frames.

These frames are still raw, uncompressed images.

---

### x264enc

```text
x264enc
```

This is the H.264 encoder.

It receives raw video frames and compresses them into the H.264 bitstream format.

Without this element, every frame would be written as raw pixel data, producing a very large file.

---

### h264parse

```text
h264parse
```

The parser analyzes and organizes the encoded H.264 stream.

Its responsibilities include:

* Formatting the stream correctly
* Extracting metadata
* Preparing the stream for storage in a container
* Improving compatibility with downstream elements

Think of it as a cleanup and packaging step between the encoder and the container.

---

### matroskamux

```text
matroskamux
```

A muxer (multiplexer) combines encoded streams into a container format.

In this case, it creates an MKV file.

The container stores:

* Video data
* Audio data (if present)
* Metadata
* Timing information

The container is not the codec.

A common beginner mistake is confusing:

| Component      | Example |
| -------------- | ------- |
| Codec          | H.264   |
| Container      | MKV     |
| File Extension | .mkv    |

The video is compressed using H.264 and then stored inside an MKV container.

---

## What Is Video Encoding?

Video encoding converts raw video frames into a compressed representation.

Raw video contains enormous amounts of information.

For example, a single 1280×720 frame contains:

```text
1280 × 720 × 3 bytes
≈ 2.6 MB
```

At 30 frames per second:

```text
2.6 MB × 30
≈ 78 MB/s
```

Storing every frame directly would quickly consume hundreds of megabytes or even gigabytes.

Video encoders reduce this size by finding redundancy between frames.

Instead of storing every pixel of every frame, the encoder stores only the information that changes.

---

## I, P, and B Frames

H.264 achieves high compression by categorizing frames into different types.

### I-Frames (Intra Frames)

An I-frame contains a complete image.

It can be decoded independently without any other frame.

Think of it as a full snapshot.

```text
Frame 1 (I)
```

---

### P-Frames (Predicted Frames)

A P-frame stores only the differences from a previous frame.

```text
Frame 1 (I)
Frame 2 (P)
Frame 3 (P)
```

If most of the image remains unchanged, the amount of stored data becomes much smaller.

---

### B-Frames (Bi-directional Frames)

A B-frame can reference both previous and future frames.

```text
Frame 1 (I)
Frame 2 (B)
Frame 3 (P)
```

This often provides even better compression efficiency.

---

## What Is H.264?

H.264, also known as **Advanced Video Coding (AVC)**, is one of the most widely used video compression standards ever created.

Its popularity comes from its ability to provide:

* High compression ratios
* Good visual quality
* Broad hardware support
* Compatibility across operating systems and devices

Although newer codecs exist, H.264 remains the default choice for many streaming, recording, and video storage applications.

---

## Other Popular Video Codecs

H.264 is not the only option.

Some common alternatives are:

| Codec        | Notes                                              |
| ------------ | -------------------------------------------------- |
| H.264 (AVC)  | Most widely supported                              |
| H.265 (HEVC) | Better compression, higher complexity              |
| VP9          | Open-source codec developed by Google              |
| AV1          | Modern codec with excellent compression efficiency |

Each codec makes different trade-offs between:

* Compression ratio
* Encoding speed
* Decoding speed
* Hardware support

---

## Software vs Hardware Encoders

Not all encoders are implemented the same way.

### Software Encoder

Our example uses:

```text
x264enc
```

This runs entirely on the CPU.

Advantages:

* Excellent quality
* Highly configurable

Disadvantages:

* Higher CPU usage
* Slower encoding

---

### Hardware Encoder

Modern GPUs and CPUs often contain dedicated video encoding hardware.

Examples include:

* NVIDIA NVENC
* Intel Quick Sync Video
* AMD Video Coding Engine (VCE)
* AMD Video Core Next (VCN)

A hardware encoder can often encode video several times faster than a software encoder while using significantly less CPU.

For NVIDIA GPUs, GStreamer provides hardware-accelerated encoders such as:

```text
nvh264enc
```

or on DeepStream platforms:

```text
nvv4l2h264enc
```

Hardware encoders are commonly used for:

* Video streaming
* Video conferencing
* Real-time recording
* AI video analytics pipelines

---

## Exercises

### Exercise 1

Generate a 10-second video at 1920×1080 resolution.

Measure the resulting file size.

```bash
video/x-raw,width=1920,height=1080
```

How much larger is it than the 1280×720 version?

---

### Exercise 2

Replace the encoder with a hardware encoder if your system supports one.

Examples:

```bash
nvh264enc
```

Compare:

* Encoding speed
* CPU usage
* File size

---

### Exercise 3

Increase the number of generated frames:

```bash
num-buffers=300
```

How does the file size change?

Is the increase proportional to the number of frames?

---

### Exercise 4

Try a different codec such as H.265 if available.

Compare:

* File size
* Encoding time
* Playback compatibility

---

## Summary

In this post, we moved from raw video storage to compressed video using H.264.

We learned:

* Why raw video files are extremely large
* How the `x264enc` element compresses video
* The role of `h264parse` and `matroskamux`
* The difference between codecs and containers
* How H.264 uses I, P, and B frames to reduce file size
* Why hardware encoders are important for real-time applications

Most real-world video pipelines use compressed formats because storing raw video is rarely practical. H.264 is often the first codec developers encounter, and understanding it provides a strong foundation for exploring more advanced codecs and streaming technologies.
