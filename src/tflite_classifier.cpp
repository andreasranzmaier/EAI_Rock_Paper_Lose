#include "tflite_classifier.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <opencv2/imgproc.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace {

constexpr int kNumThreads = 4;

int TensorElementCount(const TfLiteTensor* tensor) {
  int count = 1;
  for (int i = 0; i < tensor->dims->size; ++i) count *= tensor->dims->data[i];
  return count;
}

}  // namespace

TfliteClassifier::TfliteClassifier(const std::string& model_path) {
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (model_ == nullptr) {
    error_ = "Failed to load model: " + model_path;
    return;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  if (builder(&interpreter_) != kTfLiteOk || interpreter_ == nullptr) {
    error_ = "Failed to build interpreter.";
    return;
  }

  interpreter_->SetNumThreads(kNumThreads);

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    error_ = "Failed to allocate tensors.";
    return;
  }

  if (interpreter_->inputs().empty() || interpreter_->outputs().empty()) {
    error_ = "Model has no inputs/outputs.";
    return;
  }

  const TfLiteTensor* in = interpreter_->tensor(interpreter_->inputs()[0]);
  if (in->dims->size != 4) {
    error_ = "Expected a 4-D NHWC input tensor.";
    return;
  }
  in_h_ = in->dims->data[1];
  in_w_ = in->dims->data[2];
  in_c_ = in->dims->data[3];
  if (in_c_ != 1 && in_c_ != 3) {
    error_ = "Expected a 1-channel (grayscale) or 3-channel (RGB) input tensor.";
    return;
  }

  const TfLiteTensor* out = interpreter_->tensor(interpreter_->outputs()[0]);
  num_classes_ = TensorElementCount(out);

  ok_ = true;
}

TfliteClassifier::~TfliteClassifier() = default;

ClassifyResult TfliteClassifier::Classify(const std::uint8_t* rgb,
                                          std::size_t rgb_size) {
  ClassifyResult result;
  if (!ok_) return result;

  const int n_pixels = in_w_ * in_h_;

  // Caller always provides a full RGB frame (width * height * 3 bytes).
  if (rgb == nullptr || rgb_size < static_cast<std::size_t>(n_pixels) * 3) {
    error_ = "Input buffer too small.";
    return result;
  }

  TfLiteTensor* in = interpreter_->tensor(interpreter_->inputs()[0]);
  const float in_scale = in->params.scale;
  const int in_zp = in->params.zero_point;

  if (in_c_ == 1) {
    // Grayscale model: RGB → gray → CLAHE → Otsu binary, matching 02_process.py.
    static auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat frame(in_h_, in_w_, CV_8UC3, const_cast<std::uint8_t*>(rgb));
    cv::Mat gray, equalized, binary;
    cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    clahe->apply(gray, equalized);
    cv::threshold(equalized, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const std::uint8_t* thresh_data = binary.ptr<std::uint8_t>();

    // Optional: dump the post-CLAHE/Otsu input as PGM to inspect what the
    // model is actually seeing. Enable with: DUMP_FRAMES_DIR=/tmp/dumps ./...
    if (const char* dump_dir = std::getenv("DUMP_FRAMES_DIR")) {
      static int dump_counter = 0;
      if ((dump_counter++ % 5) == 0) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/frame_%05d.pgm", dump_dir, dump_counter);
        if (FILE* f = std::fopen(path, "wb")) {
          std::fprintf(f, "P5\n%d %d\n255\n", in_w_, in_h_);
          std::fwrite(thresh_data, 1, n_pixels, f);
          std::fclose(f);
        }
      }
    }

    switch (in->type) {
      case kTfLiteFloat32: {
        // The model contains its own Rescaling(1/255) layer, so feed raw
        // 0-255 floats - dividing here too would zero out the input.
        float* dst = interpreter_->typed_input_tensor<float>(0);
        for (int i = 0; i < n_pixels; ++i)
          dst[i] = static_cast<float>(thresh_data[i]);
        break;
      }
      case kTfLiteUInt8: {
        std::uint8_t* dst = interpreter_->typed_input_tensor<std::uint8_t>(0);
        const float scale = (in_scale > 0.0f) ? in_scale : 1.0f;
        for (int i = 0; i < n_pixels; ++i) {
          const int q = static_cast<int>(std::lround(thresh_data[i] / scale)) + in_zp;
          dst[i] = static_cast<std::uint8_t>(std::clamp(q, 0, 255));
        }
        break;
      }
      case kTfLiteInt8: {
        std::int8_t* dst = interpreter_->typed_input_tensor<std::int8_t>(0);
        const float scale = (in_scale > 0.0f) ? in_scale : 1.0f;
        for (int i = 0; i < n_pixels; ++i) {
          const int q = static_cast<int>(std::lround(thresh_data[i] / scale)) + in_zp;
          dst[i] = static_cast<std::int8_t>(std::clamp(q, -128, 127));
        }
        break;
      }
      default:
        error_ = "Unsupported input tensor type.";
        return result;
    }
  } else {
    // 3-channel RGB model: pass raw pixels through, model handles normalisation.
    const std::size_t expected = static_cast<std::size_t>(n_pixels) * 3;
    switch (in->type) {
      case kTfLiteFloat32: {
        float* dst = interpreter_->typed_input_tensor<float>(0);
        for (std::size_t i = 0; i < expected; ++i) {
          dst[i] = static_cast<float>(rgb[i]);
        }
        break;
      }
      case kTfLiteUInt8: {
        std::uint8_t* dst = interpreter_->typed_input_tensor<std::uint8_t>(0);
        if (in_scale > 0.0f) {
          for (std::size_t i = 0; i < expected; ++i) {
            const int q = static_cast<int>(std::lround(rgb[i] / in_scale)) + in_zp;
            dst[i] = static_cast<std::uint8_t>(std::clamp(q, 0, 255));
          }
        } else {
          std::memcpy(dst, rgb, expected);
        }
        break;
      }
      case kTfLiteInt8: {
        std::int8_t* dst = interpreter_->typed_input_tensor<std::int8_t>(0);
        const float scale = (in_scale > 0.0f) ? in_scale : 1.0f;
        for (std::size_t i = 0; i < expected; ++i) {
          const int q = static_cast<int>(std::lround(rgb[i] / scale)) + in_zp;
          dst[i] = static_cast<std::int8_t>(std::clamp(q, -128, 127));
        }
        break;
      }
      default:
        error_ = "Unsupported input tensor type.";
        return result;
    }
  }

  if (interpreter_->Invoke() != kTfLiteOk) {
    error_ = "Invoke failed.";
    return result;
  }

  const TfLiteTensor* out = interpreter_->tensor(interpreter_->outputs()[0]);
  result.scores.resize(num_classes_, 0.0f);

  switch (out->type) {
    case kTfLiteFloat32: {
      const float* src = interpreter_->typed_output_tensor<float>(0);
      for (int i = 0; i < num_classes_; ++i) result.scores[i] = src[i];
      break;
    }
    case kTfLiteUInt8: {
      const std::uint8_t* src = interpreter_->typed_output_tensor<std::uint8_t>(0);
      const float scale = out->params.scale;
      const int zp = out->params.zero_point;
      for (int i = 0; i < num_classes_; ++i)
        result.scores[i] = scale * (static_cast<int>(src[i]) - zp);
      break;
    }
    case kTfLiteInt8: {
      const std::int8_t* src = interpreter_->typed_output_tensor<std::int8_t>(0);
      const float scale = out->params.scale;
      const int zp = out->params.zero_point;
      for (int i = 0; i < num_classes_; ++i)
        result.scores[i] = scale * (static_cast<int>(src[i]) - zp);
      break;
    }
    default:
      error_ = "Unsupported output tensor type.";
      return result;
  }

  // The TFLite converter (TF 2.16, SavedModel path) silently drops the final
  // softmax. Apply it here so scores are interpretable probabilities. Safe to
  // do unconditionally: softmax(softmax(x)) preserves argmax.
  float max_score = result.scores[0];
  for (int i = 1; i < num_classes_; ++i)
    max_score = std::max(max_score, result.scores[i]);
  float sum_exp = 0.0f;
  for (int i = 0; i < num_classes_; ++i) {
    result.scores[i] = std::exp(result.scores[i] - max_score);
    sum_exp += result.scores[i];
  }
  for (int i = 0; i < num_classes_; ++i) result.scores[i] /= sum_exp;

  int best = 0;
  for (int i = 1; i < num_classes_; ++i) {
    if (result.scores[i] > result.scores[best]) best = i;
  }
  result.index = best;
  result.confidence = result.scores[best];
  return result;
}
