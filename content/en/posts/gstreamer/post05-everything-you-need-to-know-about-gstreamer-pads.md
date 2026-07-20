---
title: "Post05: Everything You Need to Know About Gstreamer Pads"
date: 2026-07-20T19:15:45+03:00
toc: false
images:
tags:
  - gstreamer
  - linux
---

## Introduction

Connect it to Post 04.

> In the previous article, we tried to connect `decodebin` to `videoconvert`. Unlike previous examples, the elements could not be linked immediately.
>
> To understand why, we first need to understand one of the most fundamental concepts in GStreamer: **Pads**.

---

# What is a Pad?

Explain that an element does **not** connect directly to another element.

Instead

```
  ┌─────────────────────────┐           ┌────────────────────────┐ 
  │                  Pad    │           │   Pad                  │ 
  │  Element         ┌──┐   │           │   ┌──┐        Element  │ 
  │                  │  │───┼───────────┼──►│  │                 │
  │                  └──┘   │           │   └──┘                 │ 
  │                         │           │                        │ 
  └─────────────────────────┘           └────────────────────────┘ 
```

A pad is simply a connection point.

Good analogy:

> Think of an electrical plug.
>
> A television is not connected directly to the wall.
>
> It connects through a socket.
>
> Pads play the same role for GStreamer elements.

---

# Why Pads Exist

Explain separation of concerns.

Elements only process data.

Pads transport buffers/events/queries.

Mention:

* Buffers
* Events
* Queries
* Caps negotiation

without going too deep.

---

# Source Pads

Explain

```
 ┌─────────────────────────┐ 
 │                  Src    │ 
 │  videotestsrc    ┌──┐   │ 
 │                  │  │───┼─► 
 │                  └──┘   │ 
 │                         │ 
 └─────────────────────────┘ 
```

Produces data.

---

# Sink Pads

```
 ┌────────────────────────┐ 
 │   Sink                 │ 
 │   ┌──┐  autovideosink  │ 
─┼──►│  │                 │ 
 │   └──┘                 │ 
 │                        │ 
 └────────────────────────┘ 
```

Consumes data.

---

# Linking Pads

Show

```
  src                 sink
 ┌──┐                 ┌──┐ 
 │  │────────────────►│  │ 
 └──┘                 └──┘ 
```

---

# Pad Directions

Introduce

```
GST_PAD_SRC

GST_PAD_SINK
```

---

# Pad Availability

This should be the largest section.

---

## Static Pads

Always exist.

Examples

* videotestsrc
* videoconvert
* queue
* autovideosink

Diagram

```
 ┌────────────────────────┐  
 │   sink          src    │  
 │   ┌──┐          ┌──┐   │  
 │   │  │          │  │   │  
 │   └──┘          └──┘   │  
 │     videoconvert       │  
 └────────────────────────┘  
```

---

## Sometimes Pads (Dynamic Pads)

Explain

```
decodebin
uridecodebin
rtspsrc
tsdemux
```

Diagram

Before

```
 ┌─────────────────────────┐ 
 │                         │ 
 │  decodebin              │ 
 │                         │ 
 │                         │ 
 │                         │ 
 └─────────────────────────┘
```

After

```
 ┌────────────────────────┐
 │                 src_0  │
 │ decodebin       ┌──┐   │
 │                 │  │   │
 │                 └──┘   │
 │                        │
 └────────────────────────┘
```

Explain they appear only after enough information becomes available.

---

## Request Pads

Excellent place to introduce

```
tee
```

Diagram

Without requesting

```
 ┌────────────────────────┐ 
 │ sink    tee            │ 
 │ ┌──┐                   │ 
 │ │  │                   │ 
 │ └──┘                   │ 
 └────────────────────────┘ 
```

After requesting

```
  ┌────────────────────────┐ 
  │ sink    tee      src_0 │ 
  │ ┌──┐             ┌──┐  │ 
  │ │  │             │  │  │ 
  │ └──┘             └──┘  │ 
  │                        │ 
  │                  src_1 │ 
  │                  ┌──┐  │ 
  │                  │  │  │ 
  │                  └──┘  │ 
  └────────────────────────┘ 
```

Explain why this allows unlimited branches.

---

# Ghost Pads

Introduce bins.

```
Pipeline

Bin

Element
```

Explain that ghost pads expose internal pads.

Excellent diagram.

```
+----------------------+
| Bin                  |
|                      |
| videotestsrc         |
|      │               |
| videoconvert         |
|                      |
| ghost src pad        |
+----------------------+
```

---

# Pad Templates

Introduce

```
gst-inspect-1.0
```

Show

```
Pad Templates:

SRC template:
```

Explain templates describe

* direction
* availability
* caps

without creating actual pads.

---

# Caps (Short Preview)

Don't explain deeply.

Only mention

```
video/x-raw

audio/x-raw

image/jpeg
```

Say:

> The next article will explore Caps in detail.

---

# Summary Table

| Type      | Created              | Examples     |
| --------- | -------------------- | ------------ |
| Static    | Immediately          | videoconvert |
| Sometimes | Automatically later  | decodebin    |
| Request   | Application requests | tee          |
| Ghost     | Created by bins      | Bin          |

---

# Exercises

Much stronger than coding exercises.

1. Run `gst-inspect-1.0 videoconvert`
2. Find its pad templates.
3. Inspect `decodebin`.
4. Inspect `tee`.
5. Compare pad availability.
6. Find an element with request pads.
7. Find an element with dynamic pads.

---

# Summary

Review

* Pads connect elements.
* Pads have a direction (source or sink).
* Pads have an availability (static, sometimes, or request).
* Ghost pads allow bins to expose internal elements.
* Pad templates describe pads before they exist.
