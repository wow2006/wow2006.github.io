---
title: "Part 1: Creating Your First Video File with GStreamer"
date: 2026-06-08T21:34:56+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
---

In the previous post, we displayed a test video on the screen using GStreamer. This time, we will take the next step and create our first video file.

We will start with a slightly modified version of the command from the previous article:

```bash
gst-launch-1.0 videotestsrc num-buffers=90 ! \
video/x-raw,width=640,height=480,framerate=30/1 ! \
autovideosink
```

## Understanding Caps

You may notice something new in the pipeline:

```text
video/x-raw,width=640,height=480,framerate=30/1
```

This is called **Caps** (Capabilities).

Caps describe the type of media flowing between elements. They define properties such as:

* Media type
* Resolution
* Frame rate
* Pixel format

In this example, we are requesting:

* Raw video (`video/x-raw`)
* Resolution: 640×480
* Frame rate: 30 FPS

Think of caps as a contract between elements. Every element must agree on the format of the data being exchanged.

Try changing the resolution or frame rate and observe how the pipeline behaves.

> **Note**
>
> `videotestsrc` can generate video at different resolutions and frame rates directly. In real applications, changing these properties often requires additional elements such as `videoscale` or `videorate`.

## Saving the Video to a File

Displaying video is useful, but eventually we want to save it.

Replace `autovideosink` with elements that can store the video on disk:

```bash
gst-launch-1.0 -e \
    videotestsrc num-buffers=90 ! \
    video/x-raw,width=640,height=480,framerate=30/1,format=YUY2 ! \
    matroskamux ! \
    filesink location=test.mkv
```

After the command finishes, you should see a file called:

```text
test.mkv
```

Open it with your preferred media player such as VLC.

Congratulations! You have created your first video file with GStreamer.

## Understanding the New Elements

### YUY2 Format

We extended the caps with:

```text
format=YUY2
```

YUY2 is a pixel format based on the YUV color model. It stores brightness information separately from color information and is commonly used in video capture devices.

For now, it is enough to know that it is simply another way to represent image data. We will explore pixel formats in a future article.

### Matroska Muxer

```text
matroskamux
```

A muxer combines media streams and stores them inside a container format.

`matroskamux` creates Matroska (`.mkv`) files.

### File Sink

```text
filesink location=test.mkv
```

A sink is the final destination of data in a pipeline.

Instead of displaying frames on the screen, `filesink` writes them to a file.

## Why Is the File 53 MB?

Many beginners are surprised by the file size.

The generated video contains:

* 90 frames
* Resolution: 640×480
* Format: YUY2
* No compression

Let's estimate the size.

### Size of One Frame

YUY2 uses 16 bits (2 bytes) per pixel.

```text
640 × 480 × 2
= 614,400 bytes
≈ 600 KB
```

### Size of 90 Frames

```text
614,400 × 90
= 55,296,000 bytes
≈ 52.7 MB
```

Which matches the file size we observe.

### The Important Lesson

The file is large because it contains **raw video**.

Nothing is compressed.

Every pixel of every frame is stored directly in the file.

This is similar to the difference between:

* A RAW image from a camera
* A compressed JPEG image

Raw data is large but preserves all information.

Most real-world video files use compression formats such as H.264 or H.265 to dramatically reduce file size.

For example, the same 3-second test pattern encoded with H.264 could be only a few hundred kilobytes instead of 53 MB.

This is exactly why video encoders exist.

## Visualizing the Pipeline

```text
videotestsrc
        │
        ▼
     Caps
        │
        ▼
  matroskamux
        │
        ▼
    filesink
```

The source generates frames, the caps define their format, the muxer creates an MKV container, and the file sink writes everything to disk.


You now understand one of the most important concepts in multimedia systems:

> Video size is determined not only by resolution and frame rate, but also by whether the video is compressed.

## Exercises

### Exercise 1

Generate a 1280×720 video:

```bash
gst-launch-1.0 -e \
    videotestsrc num-buffers=90 ! \
    video/x-raw,width=1280,height=720,framerate=30/1,format=YUY2 ! \
    matroskamux ! \
    filesink location=hd.mkv
```

Compare the file size with the original 640×480 version.

### Exercise 2

Change the frame rate from 30 FPS to 60 FPS while keeping 90 frames.

Observe:

* Does the file size change?
* Does the video duration change?

### Exercise 3

Generate 300 frames instead of 90.

Predict the file size before running the pipeline.

### Exercise 4

Use a different test pattern:

```bash
videotestsrc pattern=ball
```

or

```bash
videotestsrc pattern=smpte
```

Verify that the file size remains nearly identical.

## Questions

1. What is the purpose of Caps in a GStreamer pipeline?
2. What does `video/x-raw` mean?
3. Does changing the test pattern significantly affect the file size of raw video?
4. Which element in the pipeline is responsible for generating video frames?

## Summary

In this article you learned:

* What Caps are and why they are important.
* How to control video properties such as resolution and frame rate.
* How to save video data into a file.
* The purpose of `matroskamux`.
* The purpose of `filesink`.
* Why raw video files become very large.
* The difference between raw video and compressed video.
