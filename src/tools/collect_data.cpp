// Data-collection tool for the rock/paper/scissors classifier.
//
// Captures frames from the Raspberry Pi camera at full sensor resolution and
// saves them as BMP images into data/<class>/. Run it once per class, moving
// your hand through different positions / distances / lighting while it
// records, so the trained model generalises.
//
//   collect_data rock
//   collect_data paper
//   collect_data scissors
//   collect_data none       (empty scene, transitions, random poses)
//
// Press Ctrl+C to stop. Frames are saved as JPEG (q92).
//
#include "RpiCameraCapture.hpp"
#include "RgbFrameJpegExport.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

namespace {

namespace fs = std::filesystem;

// Full IMX219 sensor resolution / field of view.
constexpr unsigned int kWidth      = 3280;
constexpr unsigned int kHeight     = 2464;
constexpr int          kIntervalMs = 200;   // delay between saved frames
constexpr int          kStartDelayS = 3;    // countdown before capture starts
constexpr const char*  kOutDir     = "data";

std::atomic<bool> g_stop{false};
void HandleSignal(int) { g_stop = true; }

void SleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || argv[1][0] == '-') {
    std::cerr << "Usage: " << argv[0] << " <class>\n"
              << "  <class>  gesture label / subfolder: rock|paper|scissors|none\n";
    return 1;
  }
  const std::string class_name = argv[1];

  std::signal(SIGINT,  HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  const fs::path class_dir = fs::path(kOutDir) / class_name;
  std::error_code ec;
  fs::create_directories(class_dir, ec);
  if (ec) {
    std::cerr << "Failed to create directory " << class_dir << ": " << ec.message() << "\n";
    return 1;
  }

  // Session id groups all frames from this run, so re-running never overwrites
  // and a whole session can be held out as a test set later.
  const auto now = std::chrono::system_clock::now();
  const long long session_id =
      std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

  rpicam::CaptureParameters params;
  params.width  = kWidth;
  params.height = kHeight;
  // sensor_width/sensor_height already default to the full 3280x2464 sensor.

  std::cout << "Class '" << class_name << "' -> " << class_dir << "\n"
            << "Capturing at " << kWidth << "x" << kHeight
            << " as JPEG. Press Ctrl+C to stop.\n"
            << "Starting camera...\n";

  rpicam::RpiCameraCapture camera(params);

  // Wait for the first frame so the countdown reflects a live camera.
  std::shared_ptr<const rpicam::RgbFrame> frame;
  while (!g_stop) {
    frame = camera.currentFrame();
    if (frame) break;
    SleepMs(20);
  }
  if (g_stop) { std::cout << "Aborted before capture.\n"; return 0; }

  for (int s = kStartDelayS; s > 0 && !g_stop; --s) {
    std::cout << "Capturing in " << s << "...\n" << std::flush;
    SleepMs(1000);
  }
  std::cout << "Recording.\n";

  uint64_t last_seq = 0;
  int saved = 0;
  while (!g_stop) {
    frame = camera.currentFrame();
    if (!frame || frame->sequence == last_seq) {
      SleepMs(5);
      continue;
    }
    last_seq = frame->sequence;

    std::ostringstream name;
    name << class_name << "_" << session_id << "_"
         << std::setw(5) << std::setfill('0') << saved << ".jpg";
    const fs::path out_path = class_dir / name.str();

    try {
      rpicam::saveRgbFrameAsJpeg(frame, out_path.string());
    } catch (const std::exception& e) {
      std::cerr << "\nFailed to save " << out_path << ": " << e.what() << "\n";
      break;
    }

    ++saved;
    std::cout << "saved " << saved << "    \r" << std::flush;
    SleepMs(kIntervalMs);
  }

  std::cout << "\nDone. Saved " << saved << " frames to " << class_dir << "\n";
  return 0;
}
