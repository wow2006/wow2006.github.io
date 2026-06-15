---
title: "Post 03: Creating a video with gstreamer and C++/Rust"
date: 2026-06-15T19:21:12+03:00
toc: false
images:
series: ["gstreamer"]
tags:
  - gstreamer
  - linux
---

في المقال السابق أنشأنا ملف فيديو بصيغة {{< en >}}MKV{{< /en >}} يحتوي على فيديو مضغوط باستخدام {{< en >}}H.264{{< /en >}} باستخدام أداة {{< en >}}gst-launch-1.0{{< /en >}}.

تُعد أداة {{< en >}}gst-launch-1.0{{< /en >}} وسيلة ممتازة لتجربة خطوط المعالجة واختبارها بسرعة، لكنها ليست الطريقة المستخدمة داخل التطبيقات الحقيقية. في هذا المقال سنبني نفس خط المعالجة السابق ولكن باستخدام لغة {{< en >}}C++{{< /en >}}، وهي الخطوة الأولى نحو تطوير تطبيقات الوسائط المتعددة باستخدام {{< en >}}GStreamer{{< /en >}}.

## المتطلبات

إذا كنت قد اتبعت المقالات السابقة فمن المفترض أن معظم الحزم المطلوبة مثبتة بالفعل.

### لغة {{< en >}}C++{{< /en >}}

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

### لغة {{< en >}}Rust{{< /en >}}

```bash
sudo apt-get install rustc cargo
```

### كود البرنامج

{{< tabs >}}
{{< tab "C++" >}}
```bash
mkdir gstreamer-cpp
cd gstreamer-cpp
code main.cpp
```

تستطيع الان كتابه التالي في main.cpp

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

الان تستطيع انتاج البرنامج من ثم شغله

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

تستطيع الان كتابه التالي في main.rs

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

الان تستطيع انتاج البرنامج من ثم شغله

```bash
cargo run
```
{{< /tab >}}
{{< /tabs >}}

---

## تهيئة GStreamer

أول خطوة في أي برنامج يستخدم {{< en >}}GStreamer{{< /en >}} هي استدعاء الدالة {{< en >}}gst_init{{< /en >}}.

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


تقوم هذه الدالة بتهيئة مكتبة {{< en >}}GStreamer{{< /en >}} وتحضير المكونات الداخلية اللازمة للعمل، لذلك يجب استدعاؤها قبل استخدام أي جزء آخر من المكتبة.

## إنشاء الـ Pipeline

تُعتبر {{< en >}}Pipeline{{< /en >}} الحاوية الرئيسية التي تحتوي جميع العناصر المستخدمة في البرنامج.

في الواقع، فإن {{< en >}}GstPipeline{{< /en >}} هو نوع خاص من {{< en >}}GstBin{{< /en >}} يضيف إمكانيات إضافية مثل إدارة الحالات وإدارة الساعة وناقل الرسائل.

يمكن إنشاء {{< en >}}Pipeline{{< /en >}} جديدة باستخدام:

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

بعد ذلك نقوم بإنشاء العناصر المطلوبة كما فعلنا سابقًا باستخدام {{< en >}}gst-launch-1.0{{< /en >}}.

## إضافة العناصر إلى الـ Pipeline

بعد إنشاء العناصر يجب إضافتها إلى {{< en >}}Pipeline{{< /en >}} حتى تصبح جزءًا من خط المعالجة.

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

في هذه المرحلة أصبحت {{< en >}}Pipeline{{< /en >}} مسؤولة عن إدارة هذه العناصر، لكنها لا تعرف بعد كيفية اتصالها ببعضها البعض.

## ربط العناصر

لربط العناصر معًا نستخدم الدالة {{< en >}}gst_element_link_many{{< /en >}}.

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

تقوم هذه الدالة بربط المنافذ بين العناصر بحيث يمكن للبيانات التدفق من عنصر إلى العنصر الذي يليه.

## ضبط الخصائص

بعض العناصر توفر خصائص يمكن تعديلها أثناء الإنشاء أو قبل بدء التشغيل.

على سبيل المثال يمكن تغيير النمط المستخدم بواسطة عنصر {{< en >}}videotestsrc{{< /en >}} باستخدام:

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

تُستخدم الدالة {{< en >}}g_object_set{{< /en >}} لأن معظم عناصر {{< en >}}GStreamer{{< /en >}} مبنية على نظام الكائنات {{< en >}}GObject{{< /en >}}.

## تشغيل الـ Pipeline

بعد إنشاء جميع العناصر وربطها وضبط خصائصها، يمكن بدء التشغيل بتغيير حالة {{< en >}}Pipeline{{< /en >}} إلى {{< en >}}PLAYING{{< /en >}}.

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

عندها يبدأ تدفق البيانات داخل خط المعالجة وتنفيذ العمل المطلوب.

## انتظار الرسائل

أثناء التشغيل ترسل {{< en >}}GStreamer{{< /en >}} رسائل مختلفة عبر {{< en >}}Bus{{< /en >}} مثل:

* نهاية الملف {{< en >}}EOS{{< /en >}}
* الأخطاء {{< en >}}ERROR{{< /en >}}
* التحذيرات {{< en >}}WARNING{{< /en >}}
* تغييرات الحالة {{< en >}}STATE_CHANGED{{< /en >}}

يقوم التطبيق عادة بالاستماع إلى هذه الرسائل لمعرفة ما يحدث داخل {{< en >}}Pipeline{{< /en >}}.

لن نتعمق في {{< en >}}Bus{{< /en >}} في هذا المقال، وسنخصص له مقالًا مستقلًا لاحقًا لأنه من أهم الأجزاء في {{< en >}}GStreamer{{< /en >}}.

## الخلاصة

في هذا المقال تعرفنا على الهيكل الأساسي لأي تطبيق يستخدم {{< en >}}GStreamer{{< /en >}}:

1. تهيئة المكتبة باستخدام {{< en >}}gst_init{{< /en >}}.
2. إنشاء {{< en >}}Pipeline{{< /en >}}.
3. إنشاء العناصر المطلوبة.
4. إضافة العناصر إلى {{< en >}}Pipeline{{< /en >}}.
5. ربط العناصر معًا.
6. ضبط الخصائص عند الحاجة.
7. تشغيل {{< en >}}Pipeline{{< /en >}}.
8. متابعة الرسائل القادمة من {{< en >}}Bus{{< /en >}}.

هذه الخطوات تمثل الأساس الذي ستُبنى عليه معظم تطبيقات {{< en >}}GStreamer{{< /en >}} مهما ازدادت درجة تعقيدها.

## تمارين

### التمرين الأول

أنشئ برنامجًا يعرض فيديو الاختبار على الشاشة باستخدام خط المعالجة التالي:

```text
videotestsrc ! autovideosink
```

### التمرين الثاني

جرّب تغيير خاصية {{< en >}}pattern{{< /en >}} في عنصر {{< en >}}videotestsrc{{< /en >}} ولاحظ تأثيرها على الفيديو المعروض.

### التمرين الثالث

أنشئ ملف فيديو بصيغة {{< en >}}MKV{{< /en >}} باستخدام نفس خط المعالجة الذي استخدمناه في المقال السابق ولكن من خلال برنامج {{< en >}}C++{{< /en >}} بدلًا من {{< en >}}gst-launch-1.0{{< /en >}}.

### التمرين الرابع

أضف معالجة لرسائل الأخطاء القادمة من {{< en >}}Bus{{< /en >}} وقم بطباعة تفاصيل الخطأ على الشاشة.

