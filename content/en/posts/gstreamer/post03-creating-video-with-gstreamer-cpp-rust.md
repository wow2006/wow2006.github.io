---
title: "Post 03: Creating a video with gstreamer and C++/Rust"
date: 2026-06-15T19:22:31+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
  - cpp
  - rust
---

In the previous post, we created an MKV video file containing H.264 encoded video using `gst-launch-1.0`.

While `gst-launch-1.0` is an excellent tool for experimenting with pipelines and validating ideas, real-world applications use the GStreamer API directly. In this post, we will build the same pipeline using C++, taking our first step toward developing multimedia applications with GStreamer.

## Prerequisites

If you have followed the previous posts in this series, the runtime packages should already be installed. Calling the API from our own program needs two more things on top of that.

The first is the development headers, which is what `pkg-config` reads when we compile. The second is the plugins our pipeline actually asks for: `x264enc` lives in `plugins-ugly` and `h264parse` lives in `plugins-bad`. Post 00 described both of those as optional, and for `gst-launch-1.0` experiments they are, but this pipeline will not run without them. If an element is missing, `gst_element_factory_make` simply returns `NULL` and the program stops at `Failed to create elements`.

### C++

{{< tabs >}}
{{< tab "Ubuntu" >}}
```bash
sudo apt-get install -y gcc pkg-config curl \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly
```
{{< /tab >}}
{{< tab "Fedora" >}}
```bash
sudo dnf install -y gcc pkgconf-pkg-config curl \
    gstreamer1-devel \
    gstreamer1-plugins-base-devel \
    gstreamer1-plugins-bad-free \
    gstreamer1-plugins-ugly
```
{{< /tab >}}
{{< tab "Arch" >}}
```bash
sudo pacman -S --needed gcc pkgconf curl \
    gstreamer \
    gst-plugins-base \
    gst-plugins-bad \
    gst-plugins-ugly
```
{{< /tab >}}
{{< /tabs >}}

### Rust

The Rust bindings link against the same GStreamer development packages listed above, so install those first.

```bash
sudo apt-get install -y rustc cargo
```

## Project source code

{{< tabs >}}
{{< tab "C++" >}}
```bash
mkdir gstreamer-cpp
cd gstreamer-cpp
code main.cpp
```

You can write the following in `main.cpp`

{{< code file="gstreamer/post03/cpp/main.cpp" >}}

Now you run the code using

```bash
g++ main.cpp -o gstreamer-cpp `pkg-config --cflags --libs gstreamer-1.0`
./gstreamer-cpp
```
{{< /tab >}}
{{< tab "Rust" >}}
```bash
cargo new gstreamer-rust
cd gstreamer-rust
cargo add gstreamer
code src/main.rs
```

You can write the following in `src/main.rs`

{{< code file="gstreamer/post03/rust/src/main.rs" >}}

`cargo add gstreamer` writes the dependency for you, so your `Cargo.toml` should look like this:

{{< code file="gstreamer/post03/rust/Cargo.toml" >}}

Now you run the code using

```bash
cargo run
```
{{< /tab >}}
{{< /tabs >}}

## Initializing GStreamer

The first thing every GStreamer application must do is initialize the library by calling `gst_init`.

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post03/cpp/main.cpp" region="init" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post03/rust/src/main.rs" region="init" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

This function initializes the internal GStreamer infrastructure and prepares the library for use. It must be called before using any other GStreamer APIs.

## Creating a Pipeline

A pipeline is the top-level container that holds all elements used by an application.

Technically, `GstPipeline` is a specialized type of `GstBin` that provides additional functionality such as state management, clock management, and a message bus.

A pipeline can be created using:

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post03/cpp/main.cpp" region="pipeline" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post03/rust/src/main.rs" region="pipeline" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

Next, we create the elements required for our application, just as we did previously with `gst-launch-1.0`.

## Adding Elements to the Pipeline

Once the elements are created, they must be added to the pipeline.

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post03/cpp/main.cpp" region="add" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post03/rust/src/main.rs" region="add" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

At this point, the pipeline owns and manages these elements, but it still does not know how they should be connected.

## Linking Elements

To connect elements together, we use `gst_element_link_many`. It returns whether the whole chain was linked successfully, so it is worth checking rather than ignoring.

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post03/cpp/main.cpp" region="link" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post03/rust/src/main.rs" region="link" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

This function links the pads between elements, allowing data to flow from one element to the next.

## Configuring Element Properties

Many elements expose configurable properties that control their behavior.

Our program uses two of them. The `num-buffers` property on `videotestsrc` tells the source to produce exactly 90 frames and then stop, which at 30 frames per second gives us a three second video. The `location` property on `filesink` decides where the file is written.

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post03/cpp/main.cpp" region="properties" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post03/rust/src/main.rs" region="properties" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

Notice the difference in style between the two. In C++ we build the element first and configure it afterwards with `g_object_set`, which works because most GStreamer elements are built on top of the GObject type system. The Rust bindings instead let us set properties on the builder before the element is created, so `filesink` receives its `location` at construction time rather than in a separate step.

`videotestsrc` has many more properties than the one we use here. `pattern`, for example, changes the generated test pattern, and it is worth experimenting with.

## Starting the Pipeline

After creating, configuring, and linking all elements, we can start the pipeline by changing its state to `PLAYING`.

{{< tabs >}}
{{< tab "C++" >}}
{{< code file="gstreamer/post03/cpp/main.cpp" region="play" link="false" >}}
{{< /tab >}}
{{< tab "Rust" >}}
{{< code file="gstreamer/post03/rust/src/main.rs" region="play" link="false" >}}
{{< /tab >}}
{{< /tabs >}}

Once the pipeline enters the `PLAYING` state, data begins flowing through the pipeline and the application starts performing its intended task.

## Waiting for Messages

While running, GStreamer sends messages through a bus. Common messages include:

* `EOS` (End of Stream)
* `ERROR`
* `WARNING`
* `STATE_CHANGED`

Applications typically listen for these messages to monitor the pipeline and react to events.

We will not dive deeply into the bus in this article. It is an important topic that deserves its own dedicated post later in this series.

## Summary

In this post, we learned the basic structure of a GStreamer application:

1. Initialize GStreamer with `gst_init`.
2. Create a pipeline.
3. Create the required elements.
4. Add the elements to the pipeline.
5. Link the elements together.
6. Configure element properties when needed.
7. Start the pipeline.
8. Monitor messages through the bus.

These steps form the foundation of nearly every GStreamer application, regardless of its complexity.

## Exercises

### Exercise 1

Create a program that displays a test video using the following pipeline:

```text
videotestsrc ! autovideosink
```

### Exercise 2

Experiment with different values of the `pattern` property in `videotestsrc` and observe how the output changes.

### Exercise 3

Create an MKV video file using the same pipeline from the previous article, but this time implement it using C++ instead of `gst-launch-1.0`.

### Exercise 4

Add error handling by listening for `ERROR` messages on the bus and printing the error details to the console.
