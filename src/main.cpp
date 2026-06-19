// Rock-Paper-Lose: fair game.
//
// The Pi picks a random gesture, runs a countdown on the LED matrix, reads the
// player's gesture from the camera (requiring a stable, confident reading so a
// moving hand is not misclassified), shows its own choice for two seconds, then
// turns the whole matrix green (player won) or red (player lost).
//
// This is the FAIR version - the rigged accessory logic is added later via a
// second (accessory) model.
//
//   rock_paper_lose --model gesture.tflite
//
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "RpiCameraCapture.hpp"
#include "sense_hat_display.h"
#include "tflite_classifier.h"

namespace {

// Gesture indices used throughout: 0=rock, 1=paper, 2=scissors.
constexpr int kNumGestures = 3;
const char* const kGestureNames[kNumGestures] = {"rock", "paper", "scissors"};

// beats[g] is the gesture that g defeats: rock>scissors, paper>rock, scissors>paper.
constexpr int kBeats[kNumGestures] = {2, 0, 1};

// Stable-gesture detection: require this many consecutive frames agreeing on
// the same confident gesture before accepting it (anti-motion safeguard).
constexpr int kConsensusFrames     = 4;
constexpr float kConfidenceThresh  = 0.60f;
constexpr int kCaptureTimeoutMs    = 4000;  // give up reading a gesture after this

constexpr int kCountdownStartValue = 3;
constexpr int kCountdownStepMs     = 700;
constexpr int kShowChoiceMs        = 2000;  // show the Pi's pick for 2 s
constexpr int kBetweenRoundsMs     = 800;

std::atomic<bool> g_shutdown{false};
void HandleSignal(int) { g_shutdown = true; }

struct Options {
  std::string model_path = "gesture.tflite";
  bool use_sensehat = true;
};

void PrintUsage(const char* prog) {
  std::cerr << "Usage: " << prog << " --model gesture.tflite [--no-sensehat]\n";
}

bool ParseArgs(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") { PrintUsage(argv[0]); return false; }
    else if (a == "--no-sensehat") { o->use_sensehat = false; }
    else if (a == "--model" && i + 1 < argc) { o->model_path = argv[++i]; }
    else if (!a.empty() && a[0] != '-') { o->model_path = a; }
    else { std::cerr << "Unknown argument: " << a << "\n"; PrintUsage(argv[0]); return false; }
  }
  return true;
}

void SleepMs(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// Reads frames until a single gesture (0..2) is seen on kConsensusFrames
// consecutive confident frames, or until the timeout elapses. Returns -1 on
// timeout/shutdown.
int ReadStableGesture(const rpicam::RpiCameraCapture& camera, TfliteClassifier* classifier) {
  const std::size_t expected =
      static_cast<std::size_t>(classifier->input_width()) *
      classifier->input_height() * 3u;

  int last_class = -1;
  int streak = 0;
  uint64_t last_seq = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kCaptureTimeoutMs);

  while (!g_shutdown && std::chrono::steady_clock::now() < deadline) {
    std::shared_ptr<const rpicam::RgbFrame> frame = camera.currentFrame();
    if (!frame || frame->sequence == last_seq || frame->rgb.size() < expected) {
      SleepMs(5);
      continue;
    }
    last_seq = frame->sequence;

    const ClassifyResult r = classifier->Classify(frame->rgb.data(), frame->rgb.size());
    const bool confident_gesture = r.index >= 0 && r.index < kNumGestures && r.confidence >= kConfidenceThresh;

    if (confident_gesture && r.index == last_class) {
      if (++streak >= kConsensusFrames) return last_class;
    } else {
      last_class = confident_gesture ? r.index : -1;
      streak = confident_gesture ? 1 : 0;
    }
  }
  return -1;
}

// Returns: +1 player wins, 0 tie, -1 player loses.
int FairOutcome(int player, int pi) {
  if (player == pi) return 0;
  return (kBeats[player] == pi) ? 1 : -1;
}

int PlayRound(const rpicam::RpiCameraCapture& camera, TfliteClassifier* classifier, SenseHatDisplay* display, std::mt19937* rng) {
  std::uniform_int_distribution<int> dist(0, kNumGestures - 1);
  const int pi_choice = dist(*rng);

  // Countdown so the player presents a steady gesture at a known moment.
  for (int n = kCountdownStartValue; n > 0 && !g_shutdown; --n) {
    std::cout << "  " << n << "...\n" << std::flush;
    if (display) display->ShowDigit(n, 255, 255, 255);
    SleepMs(kCountdownStepMs);
  }
  if (g_shutdown) return 0;
  if (display) display->Clear(); // clear the countdown digits before showing the Pi's pick later

  const int player_choice = ReadStableGesture(camera, classifier);
  if (player_choice < 0) {
    std::cout << "  No stable gesture detected - replaying round.\n";
    return 0;
  }

  const int outcome = FairOutcome(player_choice, pi_choice);
  const char* outcome_str = (outcome > 0) ? "WON" : (outcome < 0) ? "LOST" : "TIE";
  std::cout << "  Pi chose " << kGestureNames[pi_choice]
            << ", player showed " << kGestureNames[player_choice]
            << " -> player " << outcome_str << ".\n";

  // Reveal the Pi's pick, coloured by the outcome: green if the player beat
  // it, red if the player lost to it, blue on a tie.
  if (display) {
    if (outcome > 0)      display->ShowGesture(pi_choice, 0, 255, 0);
    else if (outcome < 0) display->ShowGesture(pi_choice, 255, 0, 0);
    else                  display->ShowGesture(pi_choice, 0, 0, 255);
    SleepMs(kShowChoiceMs);
    display->Clear();
  }
  return outcome;
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
  if (classifier.num_classes() < kNumGestures) {
    std::cerr << "Model must output at least " << kNumGestures << " classes.\n";
    return 1;
  }
  std::cout << "Gesture model: " << classifier.input_width() << "x"
            << classifier.input_height() << ", " << classifier.num_classes()
            << " classes.\n";

  // Capture directly at the model's input resolution so frames need no resize.
  rpicam::CaptureParameters params;
  params.width = static_cast<unsigned int>(classifier.input_width());
  params.height = static_cast<unsigned int>(classifier.input_height());
  rpicam::RpiCameraCapture camera(params);

  SenseHatDisplay display;
  if (opt.use_sensehat && !display.available()) {
    std::cerr << "Sense HAT display unavailable: " << display.error_message() << "\n";
  }
  SenseHatDisplay* display_ptr =
      (opt.use_sensehat && display.available()) ? &display : nullptr;

  std::random_device rd;
  std::mt19937 rng(rd());

  std::cout << "Fair rock-paper-scissors. Press Ctrl+C to stop.\n";
  while (!g_shutdown) {
    std::cout << "New round:\n";
    PlayRound(camera, &classifier, display_ptr, &rng);
    if (!g_shutdown) SleepMs(kBetweenRoundsMs);
  }

  if (display_ptr) display_ptr->Clear();
  std::cout << "\nStopped.\n";
  return 0;
}
