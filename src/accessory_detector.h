#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// HSV-range based detector for the rigging accessory (default tuning: pink).
// Mirrors the logic in the reference Python game runner: calibrate a per-frame
// baseline count of "in-range" pixels over a short startup window, then call
// a frame an accessory hit when its in-range pixel count exceeds the baseline
// by a configurable multiplier.
//
// HSV bounds and timing live in a key=value config file so the accessory color
// can be re-tuned without rebuilding the binary.
class AccessoryDetector {
 public:
  struct Config {
    // OpenCV HSV ranges (hue 0-179, sat 0-255, val 0-255).
    int hue_min = 140;
    int hue_max = 170;
    int sat_min = 50;
    int sat_max = 255;
    int val_min = 50;
    int val_max = 255;

    // Number of frames used to establish the baseline, and how much the live
    // pixel count must exceed the baseline to count as "accessory present".
    int calibration_frames = 60;
    float threshold_multiplier = 1.25f;
  };

  AccessoryDetector() = default;

  bool LoadConfig(const std::string& path, std::string* error);
  const Config& config() const { return config_; }

  // Feed a calibration frame. Frames are accumulated until calibration_frames
  // have been seen; calibrated() then returns true.
  void Calibrate(const std::uint8_t* rgb, int width, int height);
  bool calibrated() const { return calibration_count_ >= config_.calibration_frames; }
  float baseline() const { return baseline_; }

  // Returns the in-range pixel count and whether the accessory is "present"
  // (count > baseline * threshold_multiplier). If Calibrate() has not been
  // run long enough, present is always false.
  struct CheckResult {
    int pixel_count = 0;
    bool present = false;
  };
  CheckResult Check(const std::uint8_t* rgb, int width, int height) const;

 private:
  int CountInRange(const std::uint8_t* rgb, int width, int height) const;

  Config config_{};
  int calibration_count_ = 0;
  double calibration_sum_ = 0.0;
  float baseline_ = 0.0f;
};
