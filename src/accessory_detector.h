#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// HSV-range based detector for the rigging accessory (default tuning: pink).
// Mirrors the logic from the reference Python game runner:
//   1. Calibrate a per-frame baseline count of in-HSV-range pixels over a
//      short startup window with an empty scene.
//   2. At play time, count in-range pixels per frame; the frame counts as
//      "accessory present" when its count exceeds baseline * multiplier.
// HSV bounds and timing live in a key=value config file so the accessory
// colour can be re-tuned without rebuilding the binary.
//
// Lifecycle:
//   AccessoryDetector d;
//   d.LoadConfig(path, &err);              // once, at startup
//   while (!d.calibrated()) d.Calibrate(rgb, w, h);   // ~60 frames
//   auto r = d.Check(rgb, w, h);           // every game frame
class AccessoryDetector {
 public:
  // Tunables loaded from accessory.conf. Defaults target a pink wristband
  // under typical room lighting; re-tune live with the per-frame logs.
  struct Config {
    // OpenCV HSV bounds. Hue is 0-179 (half-degrees because OpenCV stores
    // it in a uint8), sat and val are 0-255.
    int hue_min = 140;
    int hue_max = 170;
    int sat_min = 50;
    int sat_max = 255;
    int val_min = 50;
    int val_max = 255;

    // How many frames to average over when establishing the baseline.
    int calibration_frames = 60;
    // A frame is "accessory present" when its in-range pixel count exceeds
    // baseline * this multiplier. > 1.0 leaves headroom for ambient noise.
    float threshold_multiplier = 1.25f;
  };

  AccessoryDetector() = default;

  // Parses a key=value config file (see Config above). Returns false and
  // sets *error on the first malformed / unknown line; on success the new
  // values overwrite the defaults in config_.
  bool LoadConfig(const std::string& path, std::string* error);
  const Config& config() const { return config_; }

  // Feed one frame into the calibration average. No-op once calibrated() is
  // true; the caller can keep calling without harm. Frames must be RGB888,
  // top-down, width*height*3 bytes long.
  void Calibrate(const std::uint8_t* rgb, int width, int height);
  // True once calibration_frames samples have been collected; until then
  // Check() will never report present=true.
  bool calibrated() const { return calibration_count_ >= config_.calibration_frames; }
  // Average in-range pixel count seen during calibration. Mostly useful for
  // diagnostic logging at startup.
  float baseline() const { return baseline_; }

  // Per-frame result of Check(): pixel_count is the raw HSV-mask count
  // (always populated, even pre-calibration so the log can show its
  // progression), present is the boolean rigging signal.
  struct CheckResult {
    int pixel_count = 0;
    bool present = false;
  };
  // Inspect one game frame. Cheap (one cv::cvtColor + cv::inRange +
  // cv::countNonZero), so it's safe to call alongside every classifier
  // invocation.
  CheckResult Check(const std::uint8_t* rgb, int width, int height) const;

 private:
  // Shared body of Calibrate() and Check(): wraps `rgb` in a cv::Mat,
  // converts RGB->HSV, applies the configured range as a mask, returns the
  // number of pixels inside it.
  int CountInRange(const std::uint8_t* rgb, int width, int height) const;

  Config config_{};
  int calibration_count_ = 0;
  double calibration_sum_ = 0.0;  // double, not int, so 60 frames * 160*160 px = 1.5M doesn't overflow
  float baseline_ = 0.0f;
};
