#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// TF 2.16+: tflite::FlatBufferModel and tflite::Interpreter are using-aliases
// for types in tflite::impl, so forward-declare those instead of the aliases.
namespace tflite {
namespace impl {
class FlatBufferModel;
class Interpreter;
}  // namespace impl
}  // namespace tflite

// Result of a single classification.
struct ClassifyResult {
  int index = -1;               // argmax class, -1 on failure
  float confidence = 0.0f;      // score of the winning class
  std::vector<float> scores;    // per-class scores (dequantized)
};

// Thin wrapper around a single image-classification .tflite model.
//
// Contract: the caller passes raw RGB888 pixels of exactly
// input_width() * input_height() * 3 bytes (top-down, packed). The model is
// expected to bake its own rescaling/normalization into the graph (e.g.
// MobileNetV3's include_preprocessing=True), so this wrapper feeds raw
// 0-255 pixels - it only requantises when the input tensor is int8/uint8 and
// dequantises the output the same way. The classifier accepts 3-channel
// (RGB) input only; the previous 1-channel (grayscale + CLAHE/Otsu) path was
// removed when the model was switched to MobileNetV3.
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
  int num_classes() const { return num_classes_; }

  // rgb must point to input_width()*input_height()*3 bytes (top-down, packed).
  ClassifyResult Classify(const std::uint8_t* rgb, std::size_t rgb_size);

 private:
  std::unique_ptr<tflite::impl::FlatBufferModel> model_;
  std::unique_ptr<tflite::impl::Interpreter> interpreter_;

  bool ok_ = false;
  std::string error_;

  int in_w_ = 0;
  int in_h_ = 0;
  int num_classes_ = 0;
};
