// Rock-Paper-Lose: rigged game.
//
// Flow per round: countdown on the LED matrix, read the player's gesture from
// the camera (requiring a confident majority over a sliding window so a moving
// hand is not misclassified), then the Pi chooses its gesture, shows its
// symbol for two seconds, and finally turns the whole matrix green (player
// won) or red (player lost).
//
// The Pi's pick is rigged by the accessory detector (default tuning: pink):
// when the accessory is visible the Pi picks the gesture that loses to the
// player; when absent it picks the gesture that beats the player. The
// classifier and accessory checks share the same camera frames - both look at
// the player as the gesture stabilises.
//
//   rock_paper_lose --model gesture.tflite [--labels labels.txt]
//                   [--accessory-config accessory.conf] [--no-accessory]
//                   [--no-sensehat]
//
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include "RpiCameraCapture.hpp"
#include "accessory_detector.h"
#include "sense_hat_display.h"
#include "tflite_classifier.h"

namespace {

// Canonical game-side gesture indices: 0=rock, 1=paper, 2=scissors. The model
// may emit these classes at different output indices - resolved at startup
// from labels.txt.
constexpr int kNumGestures = 3;
const char* const kGestureNames[kNumGestures] = {"rock", "paper", "scissors"};

// beats[g] is the gesture that g defeats: rock>scissors, paper>rock, scissors>paper.
constexpr int kBeats[kNumGestures] = {2, 0, 1};
// Inverse: losesTo[g] is the gesture that beats g.
constexpr int kLosesTo[kNumGestures] = {1, 2, 0};

// Stable-gesture detection: a sliding window of the most recent confident
// classifications; accept the mode once the window is full and at least
// kMinAgreeing frames in it match. Sub-threshold frames are skipped (not
// added to the window) rather than resetting it, so a single blurry frame in
// the middle of a steady gesture doesn't undo the prior reads.
constexpr int kWindowSize         = 8;
constexpr int kMinAgreeing        = 5;
constexpr float kConfidenceThresh = 0.60f;
constexpr int kCaptureTimeoutMs   = 4000;

constexpr int kCountdownStartValue = 3;
constexpr int kCountdownStepMs     = 700;
constexpr int kShowChoiceMs        = 2000;  // show the Pi's pick for 2 s
constexpr int kShowResultMs        = 1500;  // green/red full screen after the pick
constexpr int kBetweenRoundsMs     = 800;

// Assignment constraints (Project_EAI.pdf):
//   - all .tflite model files combined must fit in 100 MB
//   - the application must run at >= 30 inferences/sec
// Tracked at startup (size) and per-round (inference rate) for the demo.
constexpr double kModelBudgetMB = 100.0;
constexpr double kTargetFps     = 30.0;

std::atomic<bool> g_shutdown{false};
void HandleSignal(int) { g_shutdown = true; }

struct Options {
  std::string model_path = "gesture.tflite";
  std::string labels_path;            // empty -> derive "labels.txt" next to the model
  std::string accessory_config_path;  // empty -> derive "accessory.conf" next to the model
  bool use_sensehat = true;
  bool use_accessory = true;
};

void PrintUsage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --model gesture.tflite [--labels labels.txt]"
            << " [--accessory-config accessory.conf]"
            << " [--no-accessory] [--no-sensehat]\n";
}

bool ParseArgs(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") { PrintUsage(argv[0]); return false; }
    else if (a == "--no-sensehat") { o->use_sensehat = false; }
    else if (a == "--no-accessory") { o->use_accessory = false; }
    else if (a == "--model" && i + 1 < argc) { o->model_path = argv[++i]; }
    else if (a == "--labels" && i + 1 < argc) { o->labels_path = argv[++i]; }
    else if (a == "--accessory-config" && i + 1 < argc) { o->accessory_config_path = argv[++i]; }
    else if (!a.empty() && a[0] != '-') { o->model_path = a; }
    else { std::cerr << "Unknown argument: " << a << "\n"; PrintUsage(argv[0]); return false; }
  }
  return true;
}

void SleepMs(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// Default sibling-path: same directory as the model file.
std::string SiblingPath(const std::string& model_path, const std::string& name) {
  const auto slash = model_path.find_last_of('/');
  if (slash == std::string::npos) return name;
  return model_path.substr(0, slash + 1) + name;
}

bool LoadLabels(const std::string& path, std::vector<std::string>* labels) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
      line.pop_back();
    if (!line.empty()) labels->push_back(line);
  }
  return !labels->empty();
}

// Maps each game-side gesture (0=rock,1=paper,2=scissors) to its index in the
// model's output. Returns false (and logs) if any of the three names is missing.
bool ResolveGestureIndices(const std::vector<std::string>& labels,
                           int model_idx_for_game[kNumGestures]) {
  std::unordered_map<std::string, int> by_name;
  for (int i = 0; i < static_cast<int>(labels.size()); ++i) by_name[labels[i]] = i;
  for (int g = 0; g < kNumGestures; ++g) {
    auto it = by_name.find(kGestureNames[g]);
    if (it == by_name.end()) {
      std::cerr << "labels.txt is missing class '" << kGestureNames[g] << "'.\n";
      return false;
    }
    model_idx_for_game[g] = it->second;
  }
  return true;
}

struct RoundReading {
  int gesture = -1;                 // game-side index (-1 = no stable read)
  AccessoryDetector::CheckResult accessory{};  // last accessory check, if any
};

// Polls camera frames until the sliding window of confident classifications
// reaches a majority vote, or the timeout / shutdown signal hits. The
// accessory detector (if non-null) is updated from the same frames so the
// round's rigging decision uses the player as they hold the gesture.
RoundReading ReadStableRound(const rpicam::RpiCameraCapture& camera,
                             TfliteClassifier* classifier,
                             const int model_idx_for_game[kNumGestures],
                             const std::vector<std::string>& labels,
                             AccessoryDetector* accessory) {
  RoundReading reading;
  const std::size_t expected =
      static_cast<std::size_t>(classifier->input_width()) *
      classifier->input_height() * 3u;

  // Reverse map: model output index -> game-side index, or -1 if not a gesture.
  std::vector<int> game_idx_for_model(classifier->num_classes(), -1);
  for (int g = 0; g < kNumGestures; ++g) {
    game_idx_for_model[model_idx_for_game[g]] = g;
  }

  std::deque<int> window;
  uint64_t last_seq = 0;
  const auto window_start = std::chrono::steady_clock::now();
  const auto deadline = window_start + std::chrono::milliseconds(kCaptureTimeoutMs);

  // Per-round inference perf, logged before we return so the demo can quote
  // an inferences/sec number per round (the assignment's >=30 FPS constraint).
  int inferences = 0;
  std::uint64_t classify_ns_sum = 0;
  std::uint64_t classify_ns_max = 0;
  auto LogPerf = [&]() {
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - window_start).count();
    if (inferences == 0 || elapsed_ms <= 0) return;
    const double inf_per_sec = inferences * 1000.0 / elapsed_ms;
    const double avg_ms = (classify_ns_sum / 1e6) / inferences;
    const double max_ms = classify_ns_max / 1e6;
    const char* tag = (inf_per_sec >= kTargetFps) ? "[perf]" : "[perf-warn]";
    std::cout << "  " << tag << " " << inferences << " inferences in "
              << elapsed_ms << " ms ("
              << std::fixed << std::setprecision(1) << inf_per_sec
              << " inf/s, classify avg=" << std::setprecision(2) << avg_ms
              << " ms max=" << max_ms << " ms)\n";
    std::cout.unsetf(std::ios_base::floatfield);
  };

  while (!g_shutdown && std::chrono::steady_clock::now() < deadline) {
    std::shared_ptr<const rpicam::RgbFrame> frame = camera.currentFrame();
    if (!frame || frame->sequence == last_seq || frame->rgb.size() < expected) {
      SleepMs(5);
      continue;
    }
    last_seq = frame->sequence;

    if (accessory) {
      reading.accessory =
          accessory->Check(frame->rgb.data(), classifier->input_width(),
                           classifier->input_height());
    }

    const auto t_classify_start = std::chrono::steady_clock::now();
    const ClassifyResult r = classifier->Classify(frame->rgb.data(), frame->rgb.size());
    const auto classify_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_classify_start).count());
    classify_ns_sum += classify_ns;
    if (classify_ns > classify_ns_max) classify_ns_max = classify_ns;
    ++inferences;
    const int game = (r.index >= 0 && r.index < static_cast<int>(game_idx_for_model.size()))
                         ? game_idx_for_model[r.index] : -1;
    const bool confident = (r.confidence >= kConfidenceThresh) && (game >= 0);

    const std::string& cls_name =
        (r.index >= 0 && r.index < static_cast<int>(labels.size())) ? labels[r.index] : std::string("?");
    std::cerr << "  [frame " << last_seq << "] argmax=" << cls_name
              << "(" << r.confidence << ")"
              << " window=" << window.size() << "/" << kWindowSize
              << (accessory ? std::string(" accessory_px=") + std::to_string(reading.accessory.pixel_count) : std::string(""))
              << "\n";

    // Skip - don't reset - on low confidence so a single noisy frame does
    // not undo the prior good reads in the window.
    if (!confident) continue;

    window.push_back(game);
    if (static_cast<int>(window.size()) > kWindowSize) window.pop_front();
    if (static_cast<int>(window.size()) < kWindowSize) continue;

    int votes[kNumGestures] = {0, 0, 0};
    for (int g : window) ++votes[g];
    int winner = 0;
    for (int g = 1; g < kNumGestures; ++g) if (votes[g] > votes[winner]) winner = g;
    if (votes[winner] >= kMinAgreeing) {
      reading.gesture = winner;
      LogPerf();
      return reading;
    }
  }
  LogPerf();
  return reading;
}

// Returns: +1 player wins, 0 tie, -1 player loses.
int FairOutcome(int player, int pi) {
  if (player == pi) return 0;
  return (kBeats[player] == pi) ? 1 : -1;
}

// Picks the Pi's gesture given the player's detected gesture and whether the
// rigging accessory is present.
// - accessory visible -> Pi picks the gesture the player beats (player wins).
// - accessory absent  -> Pi picks the gesture that beats the player.
// When use_accessory is false the pick is uniform random (fair game).
int ChoosePiGesture(int player_choice, bool use_accessory, bool accessory_present,
                    std::mt19937* rng) {
  if (use_accessory) {
    return accessory_present ? kBeats[player_choice] : kLosesTo[player_choice];
  }
  std::uniform_int_distribution<int> dist(0, kNumGestures - 1);
  return dist(*rng);
}

int PlayRound(const rpicam::RpiCameraCapture& camera,
              TfliteClassifier* classifier,
              const int model_idx_for_game[kNumGestures],
              const std::vector<std::string>& labels,
              SenseHatDisplay* display,
              AccessoryDetector* accessory,
              bool use_accessory,
              std::mt19937* rng) {
  for (int n = kCountdownStartValue; n > 0 && !g_shutdown; --n) {
    std::cout << "  " << n << "...\n" << std::flush;
    if (display) display->ShowDigit(n, 255, 255, 255);
    SleepMs(kCountdownStepMs);
  }
  if (g_shutdown) return 0;
  if (display) display->Clear();

  const RoundReading reading =
      ReadStableRound(camera, classifier, model_idx_for_game, labels, accessory);
  if (reading.gesture < 0) {
    std::cout << "  No stable gesture detected - replaying round.\n";
    return 0;
  }

  const int pi_choice =
      ChoosePiGesture(reading.gesture, use_accessory, reading.accessory.present, rng);
  const int outcome = FairOutcome(reading.gesture, pi_choice);
  const char* outcome_str = (outcome > 0) ? "WON" : (outcome < 0) ? "LOST" : "TIE";
  std::cout << "  Pi chose " << kGestureNames[pi_choice]
            << ", player showed " << kGestureNames[reading.gesture]
            << " (accessory=" << (reading.accessory.present ? "yes" : "no")
            << ") -> player " << outcome_str << ".\n";

  if (display) {
    display->ShowGesture(pi_choice, 255, 255, 255);
    SleepMs(kShowChoiceMs);
    if (outcome > 0)      display->ShowResult(true);
    else if (outcome < 0) display->ShowResult(false);
    else                  display->Fill(0, 0, 255);
    SleepMs(kShowResultMs);
    display->Clear();
  }
  return outcome;
}

// Runs the accessory calibration pass: pulls camera frames until the detector
// reports calibrated(). Returns false on shutdown.
bool CalibrateAccessory(const rpicam::RpiCameraCapture& camera,
                        AccessoryDetector* accessory,
                        int width, int height) {
  std::cout << "Calibrating accessory baseline ("
            << accessory->config().calibration_frames << " frames)...\n";
  uint64_t last_seq = 0;
  const std::size_t expected = static_cast<std::size_t>(width) * height * 3u;
  while (!g_shutdown && !accessory->calibrated()) {
    std::shared_ptr<const rpicam::RgbFrame> frame = camera.currentFrame();
    if (!frame || frame->sequence == last_seq || frame->rgb.size() < expected) {
      SleepMs(5);
      continue;
    }
    last_seq = frame->sequence;
    accessory->Calibrate(frame->rgb.data(), width, height);
  }
  if (g_shutdown) return false;
  std::cout << "Accessory baseline = " << accessory->baseline()
            << " px (threshold = "
            << accessory->baseline() * accessory->config().threshold_multiplier
            << " px).\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) return 1;

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  TfliteClassifier classifier(opt.model_path);
  if (!classifier.ok()) {
    std::cerr << "Failed to load gesture model: " << classifier.error_message() << "\n";
    return 1;
  }

  const std::string labels_path =
      opt.labels_path.empty() ? SiblingPath(opt.model_path, "labels.txt") : opt.labels_path;
  std::vector<std::string> labels;
  if (!LoadLabels(labels_path, &labels)) {
    std::cerr << "Failed to load labels file: " << labels_path << "\n";
    return 1;
  }
  if (static_cast<int>(labels.size()) != classifier.num_classes()) {
    std::cerr << "labels.txt has " << labels.size() << " entries but model outputs "
              << classifier.num_classes() << " classes.\n";
    return 1;
  }
  int model_idx_for_game[kNumGestures];
  if (!ResolveGestureIndices(labels, model_idx_for_game)) return 1;

  std::cout << "Gesture model: " << classifier.input_width() << "x"
            << classifier.input_height() << ", " << classifier.num_classes()
            << " classes (rock=" << model_idx_for_game[0]
            << ", paper=" << model_idx_for_game[1]
            << ", scissors=" << model_idx_for_game[2] << ").\n";

  {
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(opt.model_path, ec);
    if (!ec) {
      const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
      std::cout << "Model file: " << opt.model_path << " = "
                << std::fixed << std::setprecision(2) << mb << " MB / "
                << kModelBudgetMB << " MB budget ("
                << std::setprecision(1) << (mb / kModelBudgetMB * 100.0) << "%).\n";
      std::cout.unsetf(std::ios_base::floatfield);
    }
  }
  std::cout << "Perf target: >= " << kTargetFps
            << " inferences/sec during the gesture-read window.\n";

  AccessoryDetector accessory;
  AccessoryDetector* accessory_ptr = nullptr;
  if (opt.use_accessory) {
    const std::string accessory_path =
        opt.accessory_config_path.empty() ? "accessory.conf"
                                          : opt.accessory_config_path;
    std::string err;
    if (!accessory.LoadConfig(accessory_path, &err)) {
      std::cerr << "Accessory config: " << err
                << " - falling back to fair (random) play.\n";
      opt.use_accessory = false;
    } else {
      accessory_ptr = &accessory;
      const auto& c = accessory.config();
      std::cout << "Accessory HSV bounds: H[" << c.hue_min << "-" << c.hue_max
                << "] S[" << c.sat_min << "-" << c.sat_max
                << "] V[" << c.val_min << "-" << c.val_max
                << "], calibration_frames=" << c.calibration_frames
                << ", threshold_multiplier=" << c.threshold_multiplier << ".\n";
    }
  } else {
    std::cout << "Accessory rigging disabled - playing fair.\n";
  }

  // Capture directly at the model's input resolution so frames need no resize.
  rpicam::CaptureParameters params;
  params.width = static_cast<unsigned int>(classifier.input_width());
  params.height = static_cast<unsigned int>(classifier.input_height());
  rpicam::RpiCameraCapture camera(params);

  if (accessory_ptr && !CalibrateAccessory(camera, accessory_ptr,
                                           classifier.input_width(),
                                           classifier.input_height())) {
    return 0;
  }

  SenseHatDisplay display;
  if (opt.use_sensehat && !display.available()) {
    std::cerr << "Sense HAT display unavailable: " << display.error_message() << "\n";
  }
  SenseHatDisplay* display_ptr =
      (opt.use_sensehat && display.available()) ? &display : nullptr;

  std::random_device rd;
  std::mt19937 rng(rd());

  std::cout << "Rock-paper-lose. Press Ctrl+C to stop.\n";
  while (!g_shutdown) {
    std::cout << "New round:\n";
    PlayRound(camera, &classifier, model_idx_for_game, labels, display_ptr,
              accessory_ptr, opt.use_accessory, &rng);
    if (!g_shutdown) SleepMs(kBetweenRoundsMs);
  }

  if (display_ptr) display_ptr->Clear();
  std::cout << "\nStopped.\n";
  return 0;
}
