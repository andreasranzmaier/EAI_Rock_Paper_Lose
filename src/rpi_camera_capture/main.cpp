#include "RpiCameraCapture.hpp"
#include "RgbFrameBmpExport.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

int main()
{
    try {
        rpicam::CaptureParameters params;
        params.width = 224;
        params.height = 224;

        params.shutter_us = 0;      // set for manual shutter, e.g. 8000
        params.gain = 1.0f;         // increase for dark environments
        params.buffer_count = 6;
        params.awb = true;

        rpicam::RpiCameraCapture camera(params);

        while (true) {
            // Get current image to framebuffer.
            std::shared_ptr<const rpicam::RgbFrame> frame = camera.currentFrame();
            if (!frame) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            rpicam::saveRgbFrameAsBmp(frame, "frame.bmp");
            std::cout << "Saved frame.bmp from sequence " << frame->sequence << "\n";

            break; // store one BMP and exit
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
