// Rock-Paper-Lose: fair game.
//
// Flow per round: countdown on the LED matrix, read the player's gesture from
// the camera (requiring a stable, confident reading so a moving hand is not
// misclassified), then the Pi chooses its gesture, shows its symbol for two
// seconds, and finally turns the whole matrix green (player won) or red
// (player lost).
//
// The player gesture is read *before* the Pi picks so that the rigged
// accessory model can later choose a gesture that forces the desired outcome.
// In this fair version the Pi just rolls a die.
//
//   rock_paper_lose --model gesture.tflite [--labels labels.txt]
//
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "RpiCameraCapture.hpp"
#include "sense_hat_display.h"
#include "tflite_classifier.h"

namespace {

// Canonical game-side gesture indices: 0=rock, 1=paper, 2=scissors. The model
// may emit these classes at different output indices (e.g. alphabetical, with
// an extra "garbage" class) - resolved at startup from labels.txt.
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
constexpr int kShowResultMs        = 1500;  // green/red full screen after the pick
constexpr int kBetweenRoundsMs     = 800;

std::atomic<bool> g_shutdown{false};
void HandleSignal(int) { g_shutdown = true; }

struct Options {
  std::string model_path = "gesture.tflite";
  std::string labels_path;   // empty -> derive "labels.txt" next to the model
  bool use_sensehat = true;
};

void PrintUsage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --model gesture.tflite [--labels labels.txt] [--no-sensehat]\n";
}

bool ParseArgs(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") { PrintUsage(argv[0]); return false; }
    else if (a == "--no-sensehat") { o->use_sensehat = false; }
    else if (a == "--model" && i + 1 < argc) { o->model_path = argv[++i]; }
    else if (a == "--labels" && i + 1 < argc) { o->labels_path = argv[++i]; }
    else if (!a.empty() && a[0] != '-') { o->model_path = a; }
    else { std::cerr << "Unknown argument: " << a << "\n"; PrintUsage(argv[0]); return false; }
  }
  return true;
}

void SleepMs(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// Default labels.txt path: same directory as the model file.
std::string DefaultLabelsPath(const std::string& model_path) {
  const auto slash = model_path.find_last_of('/');
  if (slash == std::string::npos) return "labels.txt";
  return model_path.substr(0, slash + 1) + "labels.txt";
}

// Reads labels.txt (one class name per line). Trims trailing whitespace.
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

// Reads frames until a confident, stable game-side gesture (0..2) is seen on
// kConsensusFrames consecutive frames. Classes outside the rock/paper/scissors
// set (e.g. "garbage") are treated as "no confident gesture" so the consensus
// resets. Returns -1 on timeout or shutdown.
int ReadStableGesture(const rpicam::RpiCameraCapture& camera,
                      TfliteClassifier* classifier,
                      const int model_idx_for_game[kNumGestures],
                      const std::vector<std::string>& labels) {
  const std::size_t expected =
      static_cast<std::size_t>(classifier->input_width()) *
      classifier->input_height() * 3u;

  // Reverse map: model output index -> game-side index, or -1 if not a gesture.
  std::vector<int> game_idx_for_model(classifier->num_classes(), -1);
  for (int g = 0; g < kNumGestures; ++g) {
    game_idx_for_model[model_idx_for_game[g]] = g;
  }

  int last_game = -1;
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

    // Ignore the "garbage" class for now: pick argmax over rock/paper/scissors
    // only and renormalise so the confidence threshold means "wins among the
    // 3 gestures by this margin".
    float rps_sum = 0.0f;
    for (int g = 0; g < kNumGestures; ++g) rps_sum += r.scores[model_idx_for_game[g]];
    int best_game = 0;
    float best_score = r.scores[model_idx_for_game[0]];
    for (int g = 1; g < kNumGestures; ++g) {
      const float s = r.scores[model_idx_for_game[g]];
      if (s > best_score) { best_score = s; best_game = g; }
    }
    const float rps_conf = (rps_sum > 0.0f) ? best_score / rps_sum : 0.0f;
    const bool confident_gesture = rps_conf >= kConfidenceThresh;
    const int game = confident_gesture ? best_game : -1;

    const std::string& cls_name =
        (r.index >= 0 && r.index < static_cast<int>(labels.size())) ? labels[r.index] : std::string("?");
    std::cerr << "  [frame " << last_seq << "] argmax=" << cls_name << "(" << r.confidence << ")"
              << " rps_best=" << kGestureNames[best_game] << "(" << rps_conf << ")"
              << " streak=" << (confident_gesture && game == last_game ? streak + 1 : confident_gesture ? 1 : 0)
              << "\n";

    if (confident_gesture && game == last_game) {
      if (++streak >= kConsensusFrames) return last_game;
    } else {
      last_game = confident_gesture ? game : -1;
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

// Picks the Pi's gesture given the player's already-detected gesture. For now
// always random (fair game). When the accessory model is wired in, this is
// where the rigging happens: with accessory -> return kBeats[player_choice]
// inverted (Pi shows the gesture the player beats); without -> return
// kBeats[player_choice] (Pi shows the gesture that beats the player).
int ChoosePiGesture(int /*player_choice*/, std::mt19937* rng) {
  std::uniform_int_distribution<int> dist(0, kNumGestures - 1);
  return dist(*rng);
}

int PlayRound(const rpicam::RpiCameraCapture& camera,
              TfliteClassifier* classifier,
              const int model_idx_for_game[kNumGestures],
              const std::vector<std::string>& labels,
              SenseHatDisplay* display,
              std::mt19937* rng) {
  // Countdown so the player presents a steady gesture at a known moment.
  for (int n = kCountdownStartValue; n > 0 && !g_shutdown; --n) {
    std::cout << "  " << n << "...\n" << std::flush;
    if (display) display->ShowDigit(n, 255, 255, 255);
    SleepMs(kCountdownStepMs);
  }
  if (g_shutdown) return 0;
  if (display) display->Clear();

  const int player_choice = ReadStableGesture(camera, classifier, model_idx_for_game, labels);
  if (player_choice < 0) {
    std::cout << "  No stable gesture detected - replaying round.\n";
    return 0;
  }

  const int pi_choice = ChoosePiGesture(player_choice, rng);
  const int outcome = FairOutcome(player_choice, pi_choice);
  const char* outcome_str = (outcome > 0) ? "WON" : (outcome < 0) ? "LOST" : "TIE";
  std::cout << "  Pi chose " << kGestureNames[pi_choice]
            << ", player showed " << kGestureNames[player_choice]
            << " -> player " << outcome_str << ".\n";

  if (display) {
    // PDF: show the Pi's symbol for 2 s, then flash the whole matrix green
    // (won) or red (lost). Ties get a brief blue flash so the player still
    // gets feedback.
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
      opt.labels_path.empty() ? DefaultLabelsPath(opt.model_path) : opt.labels_path;
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
    PlayRound(camera, &classifier, model_idx_for_game, labels, display_ptr, &rng);
    if (!g_shutdown) SleepMs(kBetweenRoundsMs);
  }

  if (display_ptr) display_ptr->Clear();
  std::cout << "\nStopped.\n";
  return 0;
}
