#include "accessory_detector.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

#include <opencv2/imgproc.hpp>

namespace {

std::string Trim(const std::string& s) {
  std::size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  std::size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

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

int AccessoryDetector::CountInRange(const std::uint8_t* rgb, int width, int height) const {
  if (rgb == nullptr || width <= 0 || height <= 0) return 0;
  cv::Mat frame(height, width, CV_8UC3, const_cast<std::uint8_t*>(rgb));
  cv::Mat hsv, mask;
  cv::cvtColor(frame, hsv, cv::COLOR_RGB2HSV);
  cv::inRange(hsv,
              cv::Scalar(config_.hue_min, config_.sat_min, config_.val_min),
              cv::Scalar(config_.hue_max, config_.sat_max, config_.val_max),
              mask);
  return cv::countNonZero(mask);
}

void AccessoryDetector::Calibrate(const std::uint8_t* rgb, int width, int height) {
  if (calibrated()) return;
  calibration_sum_ += CountInRange(rgb, width, height);
  ++calibration_count_;
  if (calibrated()) {
    baseline_ = static_cast<float>(calibration_sum_ / calibration_count_);
  }
}

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
