use gst::prelude::*;
use gstreamer as gst;

fn main() {
    // #region init
    gst::init().unwrap();
    // #endregion

    // #region pipeline
    let pipeline = gst::Pipeline::new();
    // #endregion

    // #region properties
    let src = gst::ElementFactory::make("videotestsrc")
        .property("num-buffers", 90)
        .build()
        .expect("Failed to create videotestsrc");
    // #endregion

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

    // #region add
    pipeline
        .add_many([&src, &capsfilter, &enc, &parse, &mux, &sink])
        .unwrap();
    // #endregion

    // #region link
    gst::Element::link_many([&src, &capsfilter, &enc, &parse, &mux, &sink])
        .expect("Failed to link elements");
    // #endregion

    // #region play
    pipeline
        .set_state(gst::State::Playing)
        .expect("Unable to set pipeline to Playing");
    // #endregion

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
