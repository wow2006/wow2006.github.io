---
title: "Post 03: Creating a video with gstreamer and C++/Rust"
date: 2026-06-15T19:22:31+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
---

In the previous post, we created an MKV video file containing H.264 encoded video using `gst-launch-1.0`.

While `gst-launch-1.0` is an excellent tool for experimenting with pipelines and validating ideas, real-world applications use the GStreamer API directly. In this post, we will build the same pipeline using C++, taking our first step toward developing multimedia applications with GStreamer.

## Prerequisites

If you have followed the previous posts in this series, most of the required packages should already be installed.

### C++

{{< tabs >}}
{{< tab "Ubuntu" >}}
```bash
sudo apt-get install -y gcc pkg-config curl
```
{{< /tab >}}
{{< tab "Fedora" >}}
```bash
sudo dnf install -y gcc pkgconf-pkg-config curl
```
{{< /tab >}}
{{< tab "Arch" >}}
```bash
sudo pacman -S --needed gcc pkgconf curl
```
{{< /tab >}}
{{< /tabs >}}

### Rust

```bash
sudo apt-get install rustc cargo
```

## Project source code

{{< tabs >}}
{{< tab "C++" >}}
```bash
mkdir gstreamer-cpp
cd gstreamer-cpp
code main.cpp
```

You can write the following in `mainmain.rs`

```cpp
#include <gst/gst.h>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    GstElement *pipeline = gst_pipeline_new("test-pipeline");
    GstElement *src = gst_element_factory_make("videotestsrc", "src");
    GstElement *capsfilter = gst_element_factory_make("capsfilter", "capsfilter");
    GstElement *enc = gst_element_factory_make("x264enc", "enc");
    GstElement *parse = gst_element_factory_make("h264parse", "parse");
    GstElement *mux = gst_element_factory_make("matroskamux", "mux");
    GstElement *sink = gst_element_factory_make("filesink", "sink");

    if (!pipeline || !src || !capsfilter || !enc || !parse || !mux || !sink) {
        g_printerr("Failed to create elements\n");
        return -1;
    }

    g_object_set(src, "num-buffers", 90, nullptr);
    g_object_set(sink, "location", "test.mkv", nullptr);

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "width", G_TYPE_INT, 1280,
        "height", G_TYPE_INT, 720,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    g_object_set(capsfilter, "caps", caps, nullptr);
    gst_caps_unref(caps);

    gst_bin_add_many(GST_BIN(pipeline), src, capsfilter, enc, parse, mux, sink, nullptr);

    if (!gst_element_link_many(src, capsfilter, enc, parse, mux, sink, nullptr)) {
        g_printerr("Failed to link elements\n");
        gst_object_unref(pipeline);
        return -1;
    }

    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to set pipeline to PLAYING\n");
        gst_object_unref(pipeline);
        return -1;
    }

    GstBus *bus = gst_element_get_bus(pipeline);
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
        (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    if (msg != nullptr) {
        GError *err;
        gchar *debug_info;
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR:
                gst_message_parse_error(msg, &err, &debug_info);
                g_printerr("Error: %s\n", err->message);
                g_clear_error(&err);
                g_free(debug_info);
                break;
            case GST_MESSAGE_EOS:
                g_print("End of stream\n");
                break;
            default:
                break;
        }
        gst_message_unref(msg);
    }

    // -e flag behavior: send EOS before stopping
    gst_element_send_event(pipeline, gst_event_new_eos());

    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}
```

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

```rust
use gst::prelude::*;
use gstreamer as gst;

fn main() {
    gst::init().unwrap();

    let pipeline = gst::Pipeline::new();

    let src = gst::ElementFactory::make("videotestsrc")
        .property("num-buffers", 90)
        .build()
        .expect("Failed to create videotestsrc");

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            gst::Caps::builder("video/x-raw")
                .field("width", 1280)
                .field("height", 720)
                .field("framerate", gst::Fraction::new(30, 1))
                .build(),
        )
        .build()
        .expect("Failed to create capsfilter");

    let enc = gst::ElementFactory::make("x264enc")
        .build()
        .expect("Failed to create x264enc");

    let parse = gst::ElementFactory::make("h264parse")
        .build()
        .expect("Failed to create h264parse");

    let mux = gst::ElementFactory::make("matroskamux")
        .build()
        .expect("Failed to create matroskamux");

    let sink = gst::ElementFactory::make("filesink")
        .property("location", "test.mkv")
        .build()
        .expect("Failed to create filesink");

    pipeline
        .add_many([&src, &capsfilter, &enc, &parse, &mux, &sink])
        .unwrap();

    gst::Element::link_many([&src, &capsfilter, &enc, &parse, &mux, &sink])
        .expect("Failed to link elements");

    pipeline
        .set_state(gst::State::Playing)
        .expect("Unable to set pipeline to Playing");

    let bus = pipeline.bus().unwrap();

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;

        match msg.view() {
            MessageView::Eos(..) => {
                println!("End of stream");
                break;
            }
            MessageView::Error(err) => {
                eprintln!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            _ => (),
        }
    }

    pipeline
        .set_state(gst::State::Null)
        .expect("Unable to set pipeline to Null");
}
```

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
```cpp
gst_init(&argc, &argv);
```
{{< /tab >}}
{{< tab "Rust" >}}
```rust
gst::init().unwrap();
```
{{< /tab >}}
{{< /tabs >}}

This function initializes the internal GStreamer infrastructure and prepares the library for use. It must be called before using any other GStreamer APIs.

## Creating a Pipeline

A pipeline is the top-level container that holds all elements used by an application.

Technically, `GstPipeline` is a specialized type of `GstBin` that provides additional functionality such as state management, clock management, and a message bus.

A pipeline can be created using:

{{< tabs >}}
{{< tab "C++" >}}
```cpp
auto* pipeline = gst_pipeline_new("pipeline");
```
{{< /tab >}}
{{< tab "Rust" >}}
```rust
let pipeline = gst::Pipeline::new();
```
{{< /tab >}}
{{< /tabs >}}

Next, we create the elements required for our application, just as we did previously with `gst-launch-1.0`.

## Adding Elements to the Pipeline

Once the elements are created, they must be added to the pipeline.

{{< tabs >}}
{{< tab "C++" >}}
```cpp
gst_bin_add_many(
    GST_BIN(pipeline),
    source,
    encoder,
    muxer,
    sink,
    nullptr);
```
{{< /tab >}}
{{< tab "Rust" >}}
```rust
pipeline
    .add_many([&src, &capsfilter, &enc, &parse, &mux, &sink])
    .unwrap();
```
{{< /tab >}}
{{< /tabs >}}

At this point, the pipeline owns and manages these elements, but it still does not know how they should be connected.

## Linking Elements

To connect elements together, we use `gst_element_link_many`.

{{< tabs >}}
{{< tab "C++" >}}
```cpp
gst_element_link_many(
    source,
    encoder,
    muxer,
    sink,
    nullptr);
```
{{< /tab >}}
{{< tab "Rust" >}}
```rust
gst::Element::link_many([&src, &capsfilter, &enc, &parse, &mux, &sink])
    .expect("Failed to link elements");
```
{{< /tab >}}
{{< /tabs >}}

This function links the pads between elements, allowing data to flow from one element to the next.

## Configuring Element Properties

Many elements expose configurable properties that control their behavior.

For example, the `videotestsrc` element provides a `pattern` property that changes the generated test pattern.

{{< tabs >}}
{{< tab "C++" >}}
```cpp
g_object_set(
    source,
    "pattern",
    1,
    nullptr);
```
{{< /tab >}}
{{< tab "Rust" >}}
```rust
let src = gst::ElementFactory::make("videotestsrc")
    .property("num-buffers", 90)
    .build()
    .expect("Failed to create videotestsrc");
```
{{< /tab >}}
{{< /tabs >}}

The `g_object_set` function is used because most GStreamer elements are built on top of the GObject type system.

## Starting the Pipeline

After creating, configuring, and linking all elements, we can start the pipeline by changing its state to `PLAYING`.

{{< tabs >}}
{{< tab "C++" >}}
```cpp
gst_element_set_state(
    pipeline,
    GST_STATE_PLAYING);
```
{{< /tab >}}
{{< tab "Rust" >}}
```rust
pipeline
    .set_state(gst::State::Playing)
    .expect("Unable to set pipeline to Playing");
```
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
