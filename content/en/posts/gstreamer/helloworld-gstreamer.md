---
title: "Part 0: Helloworld Gstreamer"
date: 2026-06-05T17:54:49+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
---

In this series, we will have a beginner-friendly introduction to the world of GStreamer.

GStreamer is a powerful open-source multimedia framework used to build streaming media applications. It allows developers to create pipelines that process, transform, and transmit audio, video, and other multimedia data.

My primary focus throughout this series will be video processing. Since GStreamer is built on top of GLib, I will also highlight the differences between GStreamer APIs and GLib APIs whenever they appear. Understanding this distinction early will help avoid confusion when reading examples and documentation.

## What We Will Learn

In this article, we will cover:

* How to install GStreamer.
* How to verify that GStreamer is working correctly.
* How to use `gst-launch-1.0`.
* How to display a simple video using a GStreamer pipeline.

## Table of Contents

1. Install GStreamer.
2. Validate GStreamer Installation.
3. Display a Simple Video.
4. Exercise.

---

## 1. Install GStreamer

Before writing any code, we need to install GStreamer and its plugins.

### Ubuntu

```bash
sudo apt update

sudo apt install \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good
```

This installs:

| Package            | Description                                              |
| ------------------ | -------------------------------------------------------- |
| gstreamer1.0-tools | Command-line tools such as `gst-launch-1.0`              |
| plugins-base       | Core plugins used by most applications                   |
| plugins-good       | High-quality plugins maintained by the GStreamer project |
| plugins-bad        | Experimental or less mature plugins `You can skip it`    |
| plugins-ugly       | Plugins with licensing restrictions `You can skip it`    |

---

## 2. Validate GStreamer Installation

The simplest way to verify the installation is to check the version.

```bash
gst-launch-1.0 --version
```

Example output: (Ubuntu 26.04)

```text
gst-launch-1.0 version 1.28.2
GStreamer 1.28.2
https://launchpad.net/ubuntu/+source/gstreamer1.0
```

If you see a version number, GStreamer is installed correctly.

You can also inspect a plugin using:

```bash
gst-inspect-1.0 videotestsrc
```

This command displays information about the `videotestsrc` element.

An **element** is a building block in a GStreamer pipeline.

Examples:

* Video source
* Video decoder
* Video encoder
* Video sink
* Network source

We will discuss elements in detail in future articles.

---

## 3. Display a Simple Video

One of the most useful tools for learning GStreamer is `gst-launch-1.0`.

It allows us to create and execute pipelines directly from the command line without writing any code.

Let's display a test pattern:

```bash
gst-launch-1.0 videotestsrc ! autovideosink
```

You should see a window containing moving color bars.

The pipeline consists of two elements:

```text
videotestsrc -> autovideosink
```

Where:

* `videotestsrc` generates test video frames.
* `autovideosink` automatically selects an appropriate video output device.

The `!` character connects two elements together. It is like bash pipe.

A GStreamer pipeline is essentially a graph of connected elements.

```text
+--------------+      +---------------+
| videotestsrc | ---> | autovideosink |
+--------------+      +---------------+
```

The data flows from left to right.

### Running Forever

By default, `videotestsrc` generates frames continuously.

Press:

```text
Ctrl+C
```

to stop the pipeline.

---

## Understanding What Happened

Even though this pipeline is extremely simple, it demonstrates the core GStreamer architecture:

1. A source generates data.
2. Data flows through a pipeline.
3. A sink consumes the data.

Most real-world pipelines follow the same pattern.

```text
Source -> Processing -> Sink
```

For example:

```text
Camera -> Decoder -> AI Inference -> Display
```

or

```text
File -> Decoder -> Encoder -> Network Stream
```

As the series progresses, we will gradually replace simple elements with more advanced components.

---

## 4. Exercise

Try the following commands and observe the differences.

Display a different test pattern:

```bash
gst-launch-1.0 videotestsrc pattern=ball ! autovideosink
```

Display only 100 buffers:

```bash
gst-launch-1.0 videotestsrc num-buffers=100 ! autovideosink
```

Display detailed debugging information:

```bash
GST_DEBUG=2 gst-launch-1.0 videotestsrc ! autovideosink
```

Questions:

1. What happens when `num-buffers=100` is used?
2. How many patterns are available in `videotestsrc`?
3. What additional information appears when `GST_DEBUG=2` is enabled?

---

## Summary

In this article, we learned:

* What GStreamer is.
* How to install it.
* How to verify the installation.
* How to use `gst-launch-1.0`.
* How to create and run the simplest possible video pipeline.
