/*
// #region create-project
cargo new gstreamer-play-file-rust && cd gstreamer-play-file-rust
cargo add gstreamer
// #endregion

// #region build-run
cargo run -- big_buck_bunny_720p_h264.mov
// #endregion
*/
// #region code
use gst::prelude::*;
use gstreamer as gst;
use std::env;

fn on_pad_added(src_pad: &gst::Pad, video_sink_pad: &gst::Pad, audio_sink_pad: &gst::Pad) {
    // #region code-pad-added
    let Some(caps) = src_pad.current_caps() else {
        return;
    };
    let Some(structure) = caps.structure(0) else {
        return;
    };
    let pad_type = structure.name();

    println!("Dynamic pad created, type: {pad_type}");

    let sink_pad = if pad_type.starts_with("video/x-raw") {
        video_sink_pad
    } else if pad_type.starts_with("audio/x-raw") {
        audio_sink_pad
    } else {
        return;
    };

    if sink_pad.is_linked() {
        return;
    }

    if let Err(err) = src_pad.link(sink_pad) {
        eprintln!("Failed to link decoder pad: {err}");
    }
    // #endregion
}

fn main() {
    gst::init().unwrap();

    let args: Vec<String> = env::args().collect();
    let file_path = match args.get(1) {
        Some(path) => path.as_str(),
        None => {
            eprintln!("Usage: {} <path-to-video-file>", args[0]);
            std::process::exit(1);
        }
    };

    // #region code-create-elements
    let pipeline = gst::Pipeline::new();
    let source = gst::ElementFactory::make("filesrc")
        .property("location", file_path)
        .build()
        .expect("Failed to create filesrc");
    let decoder = gst::ElementFactory::make("decodebin")
        .build()
        .expect("Failed to create decodebin");
    let video_convert = gst::ElementFactory::make("videoconvert")
        .build()
        .expect("Failed to create videoconvert");
    let audio_convert = gst::ElementFactory::make("audioconvert")
        .build()
        .expect("Failed to create audioconvert");
    let video_sink = gst::ElementFactory::make("autovideosink")
        .build()
        .expect("Failed to create autovideosink");
    let audio_sink = gst::ElementFactory::make("autoaudiosink")
        .build()
        .expect("Failed to create autoaudiosink");
    // #endregion

    pipeline
        .add_many([&source, &decoder, &video_convert, &audio_convert, &video_sink, &audio_sink])
        .unwrap();

    // #region code-link-elements
    gst::Element::link_many([&source, &decoder]).expect("Failed to link source to decoder");
    gst::Element::link_many([&video_convert, &video_sink]).expect("Failed to link video convert to sink");
    gst::Element::link_many([&audio_convert, &audio_sink]).expect("Failed to link audio convert to sink");
    // #endregion

    // #region code-decoder-signal
    let video_sink_pad = video_convert
        .static_pad("sink")
        .expect("videoconvert has no sink pad");
    let audio_sink_pad = audio_convert
        .static_pad("sink")
        .expect("audioconvert has no sink pad");
    decoder.connect_pad_added(move |_decoder, src_pad| {
        on_pad_added(src_pad, &video_sink_pad, &audio_sink_pad);
    });
    // #endregion

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
// #endregion