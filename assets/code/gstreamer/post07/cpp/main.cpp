/*
// #region create-project
mkdir gstreamer-play-file-cpp && cd gstreamer-play-file-cpp
// #endregion
// #region build-run
g++ -std=c++20 main.cpp -o gstreamer-play-file `pkg-config --cflags --libs gstreamer-1.0`
./gstreamer-play-file big_buck_bunny_720p_h264.mov
// #endregion
*/
// #region code
#include <gst/gst.h>

#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <string_view>

namespace {

struct ElementDeleter {
  void operator()(GstElement *element) const noexcept {
    if (element)
      gst_object_unref(element);
  }
};
struct PadDeleter {
  void operator()(GstPad *pad) const noexcept {
    if (pad)
      gst_object_unref(pad);
  }
};
struct BusDeleter {
  void operator()(GstBus *bus) const noexcept {
    if (bus)
      gst_object_unref(bus);
  }
};
struct MessageDeleter {
  void operator()(GstMessage *message) const noexcept {
    if (message)
      gst_message_unref(message);
  }
};
struct CapsDeleter {
  void operator()(GstCaps *caps) const noexcept {
    if (caps)
      gst_caps_unref(caps);
  }
};

using ElementPtr = std::unique_ptr<GstElement, ElementDeleter>;
using PadPtr = std::unique_ptr<GstPad, PadDeleter>;
using BusPtr = std::unique_ptr<GstBus, BusDeleter>;
using MessagePtr = std::unique_ptr<GstMessage, MessageDeleter>;
using CapsPtr = std::unique_ptr<GstCaps, CapsDeleter>;

void on_pad_added(GstElement * /*decoder*/, GstPad *new_pad,
                  gpointer user_data) {
  auto *pipeline = static_cast<GstElement *>(user_data);

  // #region code-pad-added
  const CapsPtr new_pad_caps{gst_pad_get_current_caps(new_pad)};
  if (!new_pad_caps) {
    return;
  }

  const GstStructure *structure = gst_caps_get_structure(new_pad_caps.get(), 0);
  const std::string_view pad_type{gst_structure_get_name(structure)};

  std::cout << std::format("Dynamic pad created, type: {}\n", pad_type);

  const char *converter_name = nullptr;
  if (pad_type.starts_with("video/x-raw")) {
    converter_name = "video_convert";
  } else if (pad_type.starts_with("audio/x-raw")) {
    converter_name = "audio_convert";
  } else {
    return;
  }

  const ElementPtr converter{
      gst_bin_get_by_name(GST_BIN(pipeline), converter_name)};
  if (!converter) {
    return;
  }

  const PadPtr sink_pad{gst_element_get_static_pad(converter.get(), "sink")};
  if (!sink_pad || gst_pad_is_linked(sink_pad.get())) {
    return;
  }

  if (const GstPadLinkReturn link_result =
          gst_pad_link(new_pad, sink_pad.get());
      GST_PAD_LINK_OK != link_result) {
    std::cerr << std::format("Failed to link decoder pad, error code: {}\n",
                             static_cast<int>(link_result));
  }
  // #endregion
}

} // namespace

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << std::format("Usage: {} <path-to-video-file>\n", argv[0]);
    return EXIT_FAILURE;
  }

  // #region code-create-elements
  ElementPtr pipeline{gst_pipeline_new("play-file-pipeline")};
  ElementPtr source{gst_element_factory_make("filesrc", "source")};
  ElementPtr decoder{gst_element_factory_make("decodebin", "decoder")};
  ElementPtr video_convert{
      gst_element_factory_make("videoconvert", "video_convert")};
  ElementPtr audio_convert{
      gst_element_factory_make("audioconvert", "audio_convert")};
  ElementPtr video_sink{
      gst_element_factory_make("autovideosink", "video_sink")};
  ElementPtr audio_sink{
      gst_element_factory_make("autoaudiosink", "audio_sink")};
  // #endregion

  if (!pipeline || !source || !decoder || !video_convert || !audio_convert ||
      !video_sink || !audio_sink) {
    std::cerr << "Failed to create one or more elements\n";
    return EXIT_FAILURE;
  }

  g_object_set(source.get(), "location", argv[1], nullptr);

  GstElement *const source_raw = source.get();
  GstElement *const decoder_raw = decoder.get();
  GstElement *const video_convert_raw = video_convert.get();
  GstElement *const audio_convert_raw = audio_convert.get();
  GstElement *const video_sink_raw = video_sink.get();
  GstElement *const audio_sink_raw = audio_sink.get();

  // #region code-decoder-signal
  g_signal_connect(decoder_raw, "pad-added", G_CALLBACK(on_pad_added),
                   pipeline.get());
  // #endregion

  gst_bin_add_many(GST_BIN(pipeline.get()), source.release(), decoder.release(),
                   video_convert.release(), audio_convert.release(),
                   video_sink.release(), audio_sink.release(), nullptr);

  // #region code-link-elements
  if (!gst_element_link(source_raw, decoder_raw)) {
    std::cerr << "Failed to link filesrc to decodebin\n";
    return EXIT_FAILURE;
  }
  if (!gst_element_link(video_convert_raw, video_sink_raw)) {
    std::cerr << "Failed to link videoconvert to autovideosink\n";
    return EXIT_FAILURE;
  }
  if (!gst_element_link(audio_convert_raw, audio_sink_raw)) {
    std::cerr << "Failed to link audioconvert to autoaudiosink\n";
    return EXIT_FAILURE;
  }
  // #endregion

  if (gst_element_set_state(pipeline.get(), GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    std::cerr << "Failed to set pipeline to PLAYING\n";
    return EXIT_FAILURE;
  }

  const BusPtr bus{gst_element_get_bus(pipeline.get())};
  const MessagePtr msg{gst_bus_timed_pop_filtered(
      bus.get(), GST_CLOCK_TIME_NONE,
      static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS))};

  if (msg) {
    switch (GST_MESSAGE_TYPE(msg.get())) {
    case GST_MESSAGE_ERROR: {
      GError *err = nullptr;
      gchar *debug_info = nullptr;
      gst_message_parse_error(msg.get(), &err, &debug_info);
      std::cerr << std::format("Error: {}\n", err->message);
      g_clear_error(&err);
      g_free(debug_info);
      break;
    }
    case GST_MESSAGE_EOS:
      std::cout << "End of stream\n";
      break;
    default:
      break;
    }
  }

  gst_element_set_state(pipeline.get(), GST_STATE_NULL);
  return EXIT_SUCCESS;
}
// #endregion
