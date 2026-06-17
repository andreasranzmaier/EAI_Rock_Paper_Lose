#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rpicam {

struct CaptureParameters {
    unsigned int width = 224;
    unsigned int height = 224;

    // Used for aspect-ratio/letterbox calculation and --mode.
    // Keep these as the native sensor aspect ratio you want to preserve.
    unsigned int sensor_width = 3280;
    unsigned int sensor_height = 2464;
    unsigned int sensor_bit_depth = 8;

    int fps = 0;                 // 0 means let rpicam choose/default.
    int shutter_us = 0;          // 0 means auto exposure.
    float gain = 1.0f;           // Used when shutter_us > 0.
    int buffer_count = 6;
    bool awb = true;

    std::string rpicam_vid = "rpicam-vid";
};

struct RgbFrame {
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int stride = 0;       // width * 3

    // If output is square but the sensor is 4:3, the image is letterboxed.
    unsigned int active_x = 0;
    unsigned int active_y = 0;
    unsigned int active_width = 0;
    unsigned int active_height = 0;

    uint64_t sequence = 0;
    uint64_t capture_timestamp_ns = 0; // time when a complete YUV frame arrived
    uint64_t publish_timestamp_ns = 0; // time when RGB frame became available
    uint64_t convert_ns = 0;

    // Packed RGB888: R,G,B,R,G,B,..., top-down, no row padding.
    std::vector<uint8_t> rgb;
};

class RpiCameraCapture {
public:
    explicit RpiCameraCapture(const CaptureParameters &params = CaptureParameters{});
    ~RpiCameraCapture();

    RpiCameraCapture(const RpiCameraCapture &) = delete;
    RpiCameraCapture &operator=(const RpiCameraCapture &) = delete;

    RpiCameraCapture(RpiCameraCapture &&) = delete;
    RpiCameraCapture &operator=(RpiCameraCapture &&) = delete;

    std::shared_ptr<const RgbFrame> currentFrame() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rpicam
