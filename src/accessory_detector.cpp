// Implementation of the HSV-range accessory detector. See accessory_detector.h
// for the conceptual overview - the short version is: count pixels inside a
// configurable HSV range, calibrate a baseline from the first N frames, then
// flag a frame as "accessory present" when its in-range count comfortably
// exceeds the baseline.

#include "accessory_detector.h"

#include <cctype>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>

namespace {

// In-place would be nicer but the config keys come in once at startup, so
// allocating a trimmed copy keeps the parser simple.
std::string Trim(const std::string& s) {
  std::size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  std::size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

// Parses an int that must consume the whole string. We reject trailing junk
// (`"40 px"` or `"42abc"`) so that a typo in the config file is loud rather
// than silently truncated by std::stoi.
bool ParseInt(const std::string& s, int* out) {
  try {
    std::size_t end = 0;
    int v = std::stoi(s, &end);
    if (end != s.size()) return false;
    *out = v;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseFloat(const std::string& s, float* out) {
  try {
    std::size_t end = 0;
    float v = std::stof(s, &end);
    if (end != s.size()) return false;
    *out = v;
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

// Minimal `key = value` config reader. Each non-blank, non-`#` line must
// match one of the known keys; unknown keys / malformed values fail the load
// rather than being silently dropped, so misconfiguration shows up at startup
// instead of as mysteriously-wrong rigging during play.
bool AccessoryDetector::LoadConfig(const std::string& path, std::string* error) {
  std::ifstream in(path);
  if (!in) {
    if (error) *error = "Could not open accessory config: " + path;
    return false;
  }
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const std::string trimmed = Trim(line);
    // Skip blanks and full-line comments. (We don't bother with trailing
    // comments after a value - keep the file format dumb.)
    if (trimmed.empty() || trimmed[0] == '#') continue;
    const auto eq = trimmed.find('=');
    if (eq == std::string::npos) {
      if (error) *error = path + ":" + std::to_string(line_no) + ": expected key=value";
      return false;
    }
    const std::string key = Trim(trimmed.substr(0, eq));
    const std::string value = Trim(trimmed.substr(eq + 1));

    int iv = 0;
    float fv = 0.0f;
    bool ok = true;
    if (key == "hue_min" && ParseInt(value, &iv))                       config_.hue_min = iv;
    else if (key == "hue_max" && ParseInt(value, &iv))                  config_.hue_max = iv;
    else if (key == "sat_min" && ParseInt(value, &iv))                  config_.sat_min = iv;
    else if (key == "sat_max" && ParseInt(value, &iv))                  config_.sat_max = iv;
    else if (key == "val_min" && ParseInt(value, &iv))                  config_.val_min = iv;
    else if (key == "val_max" && ParseInt(value, &iv))                  config_.val_max = iv;
    else if (key == "calibration_frames" && ParseInt(value, &iv))       config_.calibration_frames = iv;
    else if (key == "threshold_multiplier" && ParseFloat(value, &fv))   config_.threshold_multiplier = fv;
    else                                                                ok = false;

    if (!ok) {
      if (error) *error = path + ":" + std::to_string(line_no) + ": unknown or malformed key '" + key + "'";
      return false;
    }
  }
  return true;
}

// Core HSV-mask + pixel-count primitive shared by Calibrate() and Check().
// Wrapping the caller's RGB buffer in a cv::Mat is zero-copy (we cast away
// const because cv::Mat's API demands a non-const pointer, but we never write
// through it).
int AccessoryDetector::CountInRange(const std::uint8_t* rgb, int width, int height) const {
  if (rgb == nullptr || width <= 0 || height <= 0) return 0;
  cv::Mat frame(height, width, CV_8UC3, const_cast<std::uint8_t*>(rgb));
  cv::Mat hsv, mask;
  // Convert from the camera's RGB888 to HSV so the hue/saturation bounds
  // mean what they say. Pixel ordering matters: the colleague's Python uses
  // BGR2HSV against picamera2 buffers (which are actually BGR), so the
  // numeric hue values from that setup do not transfer 1:1 here - re-tune
  // the bounds in accessory.conf if importing them.
  cv::cvtColor(frame, hsv, cv::COLOR_RGB2HSV);
  cv::inRange(hsv,
              cv::Scalar(config_.hue_min, config_.sat_min, config_.val_min),
              cv::Scalar(config_.hue_max, config_.sat_max, config_.val_max),
              mask);
  return cv::countNonZero(mask);
}

// Pull one calibration sample. Idempotent once calibrated() so calling it
// again after the buffer is full has no effect; the camera-polling loop in
// main.cpp keeps invoking until calibrated() flips to true.
void AccessoryDetector::Calibrate(const std::uint8_t* rgb, int width, int height) {
  if (calibrated()) return;
  calibration_sum_ += CountInRange(rgb, width, height);
  ++calibration_count_;
  if (calibrated()) {
    // Final sample just landed - lock in the average so Check() has a
    // baseline to compare against.
    baseline_ = static_cast<float>(calibration_sum_ / calibration_count_);
  }
}

// Per-frame query: count in-range pixels, then ask "is that comfortably
// above what we saw with an empty scene?". Returns present=false until
// calibration completes so accidental early calls cannot trigger the rigging.
AccessoryDetector::CheckResult
AccessoryDetector::Check(const std::uint8_t* rgb, int width, int height) const {
  CheckResult r;
  r.pixel_count = CountInRange(rgb, width, height);
  if (calibrated()) {
    r.present = static_cast<float>(r.pixel_count) >
                baseline_ * config_.threshold_multiplier;
  }
  return r;
}
