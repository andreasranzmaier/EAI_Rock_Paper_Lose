// Data-collection tool for the rock/paper/scissors classifier.
//
// Captures frames from the Raspberry Pi camera and saves them as BMP images
// into <out>/<class>/, one folder per gesture class. Run it once per class,
// moving your hand through different positions / distances / lighting while it
// records, so the trained model generalises.
//
//   collect_data --class rock     --count 400
//   collect_data --class paper    --count 400
//   collect_data --class scissors --count 400
//   collect_data --class none     --count 400   (empty scene, transitions, random poses)
//
#include "RpiCameraCapture.hpp"
#include "RgbFrameBmpExport.hpp"

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

std::atomic<bool> g_stop{false};
void HandleSignal(int) { g_stop = true; }

struct Options {
  std::string class_name;
  std::string out_dir = "data";
  int interval_ms     = 200;
  int count           = 0;       // 0 = capture until Ctrl+C
  int start_delay_s   = 3;
  unsigned int width  = 224;
  unsigned int height = 224;
  int shutter_us      = 0;       // 0 = auto exposure
  float gain          = 1.0f;
};

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " --class <name> [options]\n"
      << "  --class <name>      gesture label / subfolder (required): rock|paper|scissors|none\n"
      << "  --out <dir>         dataset root directory (default: data)\n"
      << "  --count <n>         frames to save, 0 = until Ctrl+C (default: 0)\n"
      << "  --interval-ms <n>   delay between saved frames (default: 200)\n"
      << "  --delay-s <n>       countdown before capture starts (default: 3)\n"
      << "  --width <n>         capture width (default: 224)\n"
      << "  --height <n>        capture height (default: 224)\n"
      << "  --shutter <us>      manual shutter in microseconds, 0 = auto (default: 0)\n"
      << "  --gain <f>          analog gain when shutter is manual (default: 1.0)\n";
}

bool ParseInt(const char* s, int* out) {
  try { *out = std::stoi(s); return true; } catch (...) { return false; }
}
bool ParseUInt(const char* s, unsigned int* out) {
  try { long v = std::stol(s); if (v < 0) return false; *out = static_cast<unsigned int>(v); return true; }
  catch (...) { return false; }
}
bool ParseFloat(const char* s, float* out) {
  try { *out = std::stof(s); return true; } catch (...) { return false; }
}

bool ParseArgs(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::cerr << name << " needs a value\n"; return nullptr; }
      return argv[++i];
    };
    if (a == "-h" || a == "--help")    { PrintUsage(argv[0]); return false; }
    else if (a == "--class")           { const char* v = need("--class");       if (!v) return false; o->class_name = v; }
    else if (a == "--out")             { const char* v = need("--out");         if (!v) return false; o->out_dir = v; }
    else if (a == "--count")           { const char* v = need("--count");       if (!v || !ParseInt(v, &o->count)) return false; }
    else if (a == "--interval-ms")     { const char* v = need("--interval-ms"); if (!v || !ParseInt(v, &o->interval_ms)) return false; }
    else if (a == "--delay-s")         { const char* v = need("--delay-s");     if (!v || !ParseInt(v, &o->start_delay_s)) return false; }
    else if (a == "--width")           { const char* v = need("--width");       if (!v || !ParseUInt(v, &o->width)) return false; }
    else if (a == "--height")          { const char* v = need("--height");      if (!v || !ParseUInt(v, &o->height)) return false; }
    else if (a == "--shutter")         { const char* v = need("--shutter");     if (!v || !ParseInt(v, &o->shutter_us)) return false; }
    else if (a == "--gain")            { const char* v = need("--gain");        if (!v || !ParseFloat(v, &o->gain)) return false; }
    else { std::cerr << "Unknown argument: " << a << "\n"; PrintUsage(argv[0]); return false; }
  }
  if (o->class_name.empty()) { std::cerr << "--class is required.\n"; PrintUsage(argv[0]); return false; }
  if (o->interval_ms < 0) o->interval_ms = 0;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) return 1;

  std::signal(SIGINT,  HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  const fs::path class_dir = fs::path(opt.out_dir) / opt.class_name;
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
  params.width      = opt.width;
  params.height     = opt.height;
  params.shutter_us = opt.shutter_us;
  params.gain       = opt.gain;

  std::cout << "Class '" << opt.class_name << "' -> " << class_dir << "\n"
            << "Starting camera (" << opt.width << "x" << opt.height << ")...\n";

  rpicam::RpiCameraCapture camera(params);

  // Wait for the first frame so the countdown reflects a live camera.
  std::shared_ptr<const rpicam::RgbFrame> frame;
  while (!g_stop) {
    frame = camera.currentFrame();
    if (frame) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (g_stop) { std::cout << "Aborted before capture.\n"; return 0; }

  for (int s = opt.start_delay_s; s > 0 && !g_stop; --s) {
    std::cout << "Capturing in " << s << "...\n" << std::flush;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  std::cout << "Recording. Press Ctrl+C to stop.\n";

  uint64_t last_seq = 0;
  int saved = 0;
  while (!g_stop && (opt.count == 0 || saved < opt.count)) {
    frame = camera.currentFrame();
    if (!frame || frame->sequence == last_seq) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    last_seq = frame->sequence;

    std::ostringstream name;
    name << opt.class_name << "_" << session_id << "_"
         << std::setw(5) << std::setfill('0') << saved << ".bmp";
    const fs::path out_path = class_dir / name.str();

    try {
      rpicam::saveRgbFrameAsBmp(frame, out_path.string());
    } catch (const std::exception& e) {
      std::cerr << "\nFailed to save " << out_path << ": " << e.what() << "\n";
      break;
    }

    ++saved;
    std::cout << "saved " << saved
              << (opt.count ? "/" + std::to_string(opt.count) : "")
              << "    \r" << std::flush;

    if (opt.interval_ms > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(opt.interval_ms));
  }

  std::cout << "\nDone. Saved " << saved << " frames to " << class_dir << "\n";
  return 0;
}
