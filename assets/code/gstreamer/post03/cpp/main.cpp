#include <gst/gst.h>

int main(int argc, char *argv[]) {
    // #region init
    gst_init(&argc, &argv);
    // #endregion

    // #region pipeline
    GstElement *pipeline = gst_pipeline_new("test-pipeline");
    // #endregion

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

    // #region properties
    g_object_set(src, "num-buffers", 90, nullptr);
    g_object_set(sink, "location", "test.mkv", nullptr);
    // #endregion

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "width", G_TYPE_INT, 1280,
        "height", G_TYPE_INT, 720,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    g_object_set(capsfilter, "caps", caps, nullptr);
    gst_caps_unref(caps);

    // #region add
    gst_bin_add_many(GST_BIN(pipeline), src, capsfilter, enc, parse, mux, sink, nullptr);
    // #endregion

    // #region link
    if (!gst_element_link_many(src, capsfilter, enc, parse, mux, sink, nullptr)) {
        g_printerr("Failed to link elements\n");
        gst_object_unref(pipeline);
        return -1;
    }
    // #endregion

    // #region play
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to set pipeline to PLAYING\n");
        gst_object_unref(pipeline);
        return -1;
    }
    // #endregion

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

    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}
