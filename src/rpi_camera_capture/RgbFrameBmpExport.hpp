#pragma once

#include "RpiCameraCapture.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rpicam {
namespace bmp_export_detail {

inline void writeU16Le(std::ofstream &out, uint16_t value)
{
    out.put(static_cast<char>(value & 0xffu));
    out.put(static_cast<char>((value >> 8u) & 0xffu));
}

inline void writeU32Le(std::ofstream &out, uint32_t value)
{
    out.put(static_cast<char>(value & 0xffu));
    out.put(static_cast<char>((value >> 8u) & 0xffu));
    out.put(static_cast<char>((value >> 16u) & 0xffu));
    out.put(static_cast<char>((value >> 24u) & 0xffu));
}

inline void writeI32Le(std::ofstream &out, int32_t value)
{
    writeU32Le(out, static_cast<uint32_t>(value));
}

} // namespace bmp_export_detail

// Minimal BMP export for RgbFrame.
// Input:  packed RGB888, top-down, frame.stride bytes per row.
// Output: standard 24-bit BMP, bottom-up, BGR rows, 4-byte row padding.
inline void saveRgbFrameAsBmp(const RgbFrame &frame, const std::string &filename)
{
    if (frame.width == 0 || frame.height == 0) {
        throw std::runtime_error("saveRgbFrameAsBmp: invalid frame size");
    }

    const uint32_t width = frame.width;
    const uint32_t height = frame.height;
    const uint32_t input_stride = frame.stride != 0 ? frame.stride : width * 3u;

    if (input_stride < width * 3u) {
        throw std::runtime_error("saveRgbFrameAsBmp: invalid RGB stride");
    }

    const size_t required_size = static_cast<size_t>(input_stride) * height;
    if (frame.rgb.size() < required_size) {
        throw std::runtime_error("saveRgbFrameAsBmp: RGB buffer is smaller than expected");
    }

    const uint32_t bmp_row_size = (width * 3u + 3u) & ~3u;
    const uint32_t pixel_data_size = bmp_row_size * height;

    constexpr uint32_t file_header_size = 14u;
    constexpr uint32_t dib_header_size = 40u;
    constexpr uint32_t pixel_data_offset = file_header_size + dib_header_size;
    const uint32_t file_size = pixel_data_offset + pixel_data_size;

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("saveRgbFrameAsBmp: failed to open output file: " + filename);
    }

    using namespace bmp_export_detail;

    // BMP file header.
    out.put('B');
    out.put('M');
    writeU32Le(out, file_size);
    writeU16Le(out, 0);
    writeU16Le(out, 0);
    writeU32Le(out, pixel_data_offset);

    // BITMAPINFOHEADER.
    writeU32Le(out, dib_header_size);
    writeI32Le(out, static_cast<int32_t>(width));
    writeI32Le(out, static_cast<int32_t>(height)); // positive means bottom-up BMP
    writeU16Le(out, 1);                            // planes
    writeU16Le(out, 24);                           // bits per pixel
    writeU32Le(out, 0);                            // BI_RGB, no compression
    writeU32Le(out, pixel_data_size);
    writeI32Le(out, 2835);                         // approx. 72 DPI
    writeI32Le(out, 2835);
    writeU32Le(out, 0);
    writeU32Le(out, 0);

    std::vector<uint8_t> row(bmp_row_size, 0);

    // Source is top-down RGB. BMP stores bottom-up BGR.
    for (int y = static_cast<int>(height) - 1; y >= 0; --y) {
        const uint8_t *src = frame.rgb.data() + static_cast<size_t>(y) * input_stride;
        std::fill(row.begin(), row.end(), 0);

        for (uint32_t x = 0; x < width; ++x) {
            row[x * 3u + 0u] = src[x * 3u + 2u]; // B
            row[x * 3u + 1u] = src[x * 3u + 1u]; // G
            row[x * 3u + 2u] = src[x * 3u + 0u]; // R
        }

        out.write(reinterpret_cast<const char *>(row.data()),
                  static_cast<std::streamsize>(row.size()));
    }

    if (!out) {
        throw std::runtime_error("saveRgbFrameAsBmp: failed while writing: " + filename);
    }
}

inline void saveRgbFrameAsBmp(const std::shared_ptr<const RgbFrame> &frame,
                              const std::string &filename)
{
    if (!frame) {
        throw std::runtime_error("saveRgbFrameAsBmp: frame is null");
    }

    saveRgbFrameAsBmp(*frame, filename);
}

} // namespace rpicam
