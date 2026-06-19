#pragma once

#include "RpiCameraCapture.hpp"

#include <jpeglib.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace rpicam {

// Encodes an RgbFrame (packed RGB888, top-down) to a JPEG file using libjpeg.
// quality is 1..100 (higher = better quality, larger file). 92 is a good
// default for training data: visually lossless, ~20x smaller than BMP.
inline void saveRgbFrameAsJpeg(const RgbFrame &frame, const std::string &filename, int quality = 92)
{
    if (frame.width == 0 || frame.height == 0) {
        throw std::runtime_error("saveRgbFrameAsJpeg: invalid frame size");
    }

    const unsigned int stride = frame.stride != 0 ? frame.stride : frame.width * 3u;
    if (stride < frame.width * 3u) {
        throw std::runtime_error("saveRgbFrameAsJpeg: invalid RGB stride");
    }
    if (frame.rgb.size() < static_cast<size_t>(stride) * frame.height) {
        throw std::runtime_error("saveRgbFrameAsJpeg: RGB buffer smaller than expected");
    }

    FILE *outfile = std::fopen(filename.c_str(), "wb");
    if (outfile == nullptr) {
        throw std::runtime_error("saveRgbFrameAsJpeg: failed to open output file: " + filename);
    }

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = frame.width;
    cinfo.image_height = frame.height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        // libjpeg does not modify the input, so the const_cast is safe.
        JSAMPROW row = const_cast<JSAMPROW>(
            frame.rgb.data() + static_cast<size_t>(cinfo.next_scanline) * stride);
        jpeg_write_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(outfile);
}

inline void saveRgbFrameAsJpeg(const std::shared_ptr<const RgbFrame> &frame, const std::string &filename, int quality = 92)
{
    if (!frame) {
        throw std::runtime_error("saveRgbFrameAsJpeg: frame is null");
    }
    saveRgbFrameAsJpeg(*frame, filename, quality);
}

} // namespace rpicam
