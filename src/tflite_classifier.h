#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tflite {
class FlatBufferModel;
class Interpreter;
}  // namespace tflite

// Result of a single classification.
struct ClassifyResult {
  int index = -1;               // argmax class, -1 on failure
  float confidence = 0.0f;      // score of the winning class
  std::vector<float> scores;    // per-class scores (dequantized)
};

// Generic image classifier around a single .tflite model. Used for both the
// gesture model (rock/paper/scissors[/none]) and the accessory model
// (accessory / no-accessory) - they differ only in the model file.
//
// Contract: the caller passes raw RGB888 pixels sized exactly
// input_width() * input_height() * 3. The model is expected to include its own
// rescaling/normalization as its first layer, so this wrapper feeds raw 0-255
// pixels (quantized according to the input tensor when the model is int8/uint8).
class TfliteClassifier {
 public:
  explicit TfliteClassifier(const std::string& model_path);
  ~TfliteClassifier();

  TfliteClassifier(const TfliteClassifier&) = delete;
  TfliteClassifier& operator=(const TfliteClassifier&) = delete;

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_; }

  int input_width() const { return in_w_; }
  int input_height() const { return in_h_; }
  int input_channels() const { return in_c_; }
  int num_classes() const { return num_classes_; }

  // rgb must point to input_width()*input_height()*3 bytes (top-down, packed).
  ClassifyResult Classify(const std::uint8_t* rgb, std::size_t rgb_size);

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

  bool ok_ = false;
  std::string error_;

  int in_w_ = 0;
  int in_h_ = 0;
  int in_c_ = 0;
  int num_classes_ = 0;
};
