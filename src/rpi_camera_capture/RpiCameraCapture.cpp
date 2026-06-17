#include "RpiCameraCapture.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace rpicam {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

unsigned int evenAtLeast(unsigned int value, unsigned int minimum)
{
    value = std::max(value, minimum);
    if (value & 1u)
        --value;
    return std::max(value, minimum);
}

unsigned int alignUp(unsigned int value, unsigned int alignment)
{
    if (alignment == 0)
        return value;

    return ((value + alignment - 1u) / alignment) * alignment;
}

CaptureParameters sanitize(CaptureParameters p)
{
    p.width = evenAtLeast(p.width, 2);
    p.height = evenAtLeast(p.height, 2);
    p.sensor_width = evenAtLeast(p.sensor_width, 2);
    p.sensor_height = evenAtLeast(p.sensor_height, 2);
    p.fps = std::max(0, p.fps);
    p.shutter_us = std::max(0, p.shutter_us);
    p.gain = std::max(1.0f, p.gain);
    p.buffer_count = std::max(2, p.buffer_count);
    if (p.sensor_bit_depth == 0)
        p.sensor_bit_depth = 8;
    if (p.rpicam_vid.empty())
        p.rpicam_vid = "rpicam-vid";
    return p;
}

struct Layout {
    unsigned int out_w = 0;
    unsigned int out_h = 0;
    unsigned int active_x = 0;
    unsigned int active_y = 0;
    unsigned int active_w = 0;
    unsigned int active_h = 0;
    unsigned int stream_w = 0;
    unsigned int stream_h = 0;
};

Layout computeLayout(const CaptureParameters &p)
{
    Layout l;
    l.out_w = p.width;
    l.out_h = p.height;

    const double sensor_aspect = static_cast<double>(p.sensor_width) / p.sensor_height;
    const double output_aspect = static_cast<double>(p.width) / p.height;

    if (std::fabs(sensor_aspect - output_aspect) < 0.003) {
        l.active_w = p.width;
        l.active_h = p.height;
    } else if (output_aspect < sensor_aspect) {
        // Output is too tall/narrow. Keep full width and letterbox top/bottom.
        l.active_w = p.width;
        l.active_h = evenAtLeast(static_cast<unsigned int>(std::lround(p.width / sensor_aspect)), 2);
        if (l.active_h > p.height)
            l.active_h = p.height;
        l.active_y = (p.height - l.active_h) / 2u;
    } else {
        // Output is too wide. Keep full height and pillarbox left/right.
        l.active_h = p.height;
        l.active_w = evenAtLeast(static_cast<unsigned int>(std::lround(p.height * sensor_aspect)), 2);
        if (l.active_w > p.width)
            l.active_w = p.width;
        l.active_x = (p.width - l.active_w) / 2u;
    }

    // IMPORTANT:
    // Raw YUV420 output from rpicam-vid is sensitive to width alignment.
    // Requesting a non-aligned width such as 224 can cause the consumer to read
    // the wrong frame boundaries, producing striped/corrupted RGB/BMP output.
    // 128-byte width alignment is the important fix here.
    constexpr unsigned int yuv420_width_alignment = 128;

    l.stream_w = alignUp(evenAtLeast(l.active_w, 2), yuv420_width_alignment);

    // Preserve the same sensor aspect ratio in the raw stream, then scale the
    // raw stream down into the active area of the requested RGB output.
    l.stream_h = evenAtLeast(
        static_cast<unsigned int>(std::lround(l.stream_w / sensor_aspect)),
        2);

    return l;
}

std::string toString(unsigned int v)
{
    return std::to_string(v);
}

std::string toString(int v)
{
    return std::to_string(v);
}

std::string toString(float v)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(v));
    return std::string(buf);
}

int clampByte(int v)
{
    return std::min(255, std::max(0, v));
}

bool readExact(int fd, uint8_t *dst, size_t bytes, const std::atomic<bool> &stop_requested)
{
    size_t done = 0;
    while (done < bytes && !stop_requested.load(std::memory_order_acquire)) {
        const ssize_t r = ::read(fd, dst + done, bytes - done);
        if (r > 0) {
            done += static_cast<size_t>(r);
            continue;
        }
        if (r == 0)
            return false;
        if (errno == EINTR)
            continue;
        return false;
    }
    return done == bytes;
}

void yuv420ToLetterboxedRgb(const std::vector<uint8_t> &yuv,
                            const Layout &l,
                            std::vector<uint8_t> &rgb)
{
    rgb.assign(static_cast<size_t>(l.out_w) * l.out_h * 3u, 0);

    const uint8_t *y_plane = yuv.data();
    const uint8_t *u_plane = y_plane + static_cast<size_t>(l.stream_w) * l.stream_h;
    const uint8_t *v_plane = u_plane + static_cast<size_t>(l.stream_w / 2u) * (l.stream_h / 2u);

    for (unsigned int oy = 0; oy < l.active_h; ++oy) {
        const unsigned int sy = static_cast<unsigned int>(
            (static_cast<uint64_t>(oy) * l.stream_h) / l.active_h);
        const unsigned int sy_uv = sy / 2u;

        uint8_t *out = rgb.data() +
            (static_cast<size_t>(l.active_y + oy) * l.out_w + l.active_x) * 3u;

        for (unsigned int ox = 0; ox < l.active_w; ++ox) {
            const unsigned int sx = static_cast<unsigned int>(
                (static_cast<uint64_t>(ox) * l.stream_w) / l.active_w);
            const unsigned int sx_uv = sx / 2u;

            const int y = static_cast<int>(y_plane[static_cast<size_t>(sy) * l.stream_w + sx]);
            const int u = static_cast<int>(u_plane[static_cast<size_t>(sy_uv) * (l.stream_w / 2u) + sx_uv]) - 128;
            const int v = static_cast<int>(v_plane[static_cast<size_t>(sy_uv) * (l.stream_w / 2u) + sx_uv]) - 128;

            // BT.601-ish limited-range YUV to RGB.
            const int c = std::max(0, y - 16);
            const int r = (298 * c + 409 * v + 128) >> 8;
            const int g = (298 * c - 100 * u - 208 * v + 128) >> 8;
            const int b = (298 * c + 516 * u + 128) >> 8;

            out[ox * 3u + 0u] = static_cast<uint8_t>(clampByte(r));
            out[ox * 3u + 1u] = static_cast<uint8_t>(clampByte(g));
            out[ox * 3u + 2u] = static_cast<uint8_t>(clampByte(b));
        }
    }
}

} // namespace

class RpiCameraCapture::Impl {
public:
    explicit Impl(const CaptureParameters &params)
        : params_(sanitize(params)), layout_(computeLayout(params_)), thread_(&Impl::threadMain, this)
    {
    }

    ~Impl()
    {
        stop_requested_.store(true, std::memory_order_release);
        const pid_t pid = child_pid_.load(std::memory_order_acquire);
        if (pid > 0)
            ::kill(pid, SIGTERM);
        if (thread_.joinable())
            thread_.join();
    }

    std::shared_ptr<const RgbFrame> currentFrame() const
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return latest_;
    }

private:
    std::vector<std::string> command() const
    {
        std::vector<std::string> args;
        args.push_back(params_.rpicam_vid);
        args.push_back("--nopreview");
        args.push_back("--timeout");
        args.push_back("0");
        args.push_back("--codec");
        args.push_back("yuv420");
        args.push_back("--output");
        args.push_back("-");
        args.push_back("--width");
        args.push_back(toString(layout_.stream_w));
        args.push_back("--height");
        args.push_back(toString(layout_.stream_h));
        args.push_back("--mode");
        args.push_back(toString(params_.sensor_width) + ":" +
                       toString(params_.sensor_height) + ":" +
                       toString(params_.sensor_bit_depth));
        args.push_back("--buffer-count");
        args.push_back(toString(params_.buffer_count));
        args.push_back("--denoise");
        args.push_back("cdn_off");

        if (params_.fps > 0) {
            args.push_back("--framerate");
            args.push_back(toString(params_.fps));
        }
        if (params_.shutter_us > 0) {
            args.push_back("--shutter");
            args.push_back(toString(params_.shutter_us));
            args.push_back("--gain");
            args.push_back(toString(params_.gain));
        }
        if (!params_.awb) {
            args.push_back("--awbgains");
            args.push_back("1,1");
        }
        return args;
    }

    int startRpicamProcess()
    {
        std::vector<std::string> args = command();
        std::vector<char *> argv;
        argv.reserve(args.size() + 1u);
        for (std::string &s : args)
            argv.push_back(const_cast<char *>(s.c_str()));
        argv.push_back(nullptr);

        int pipefd[2];
        if (::pipe(pipefd) != 0)
            throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));

        const pid_t pid = ::fork();
        if (pid < 0) {
            ::close(pipefd[0]);
            ::close(pipefd[1]);
            throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
        }

        if (pid == 0) {
            ::dup2(pipefd[1], STDOUT_FILENO);
            ::close(pipefd[0]);
            ::close(pipefd[1]);
            ::execvp(argv[0], argv.data());
            _exit(127);
        }

        ::close(pipefd[1]);
        child_pid_.store(pid, std::memory_order_release);

        std::cerr << "started:";
        for (const std::string &s : args)
            std::cerr << " " << s;
        std::cerr << "\n";

        std::cerr << "capture layout: output=" << layout_.out_w << "x" << layout_.out_h
                  << " active=" << layout_.active_w << "x" << layout_.active_h
                  << " at " << layout_.active_x << "," << layout_.active_y
                  << " stream=" << layout_.stream_w << "x" << layout_.stream_h
                  << "\n";

        return pipefd[0];
    }

    void stopRpicamProcess()
    {
        const pid_t pid = child_pid_.exchange(-1, std::memory_order_acq_rel);
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            int status = 0;
            while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }

    void threadMain()
    {
        int fd = -1;
        try {
            fd = startRpicamProcess();

            const size_t y_size = static_cast<size_t>(layout_.stream_w) * layout_.stream_h;
            const size_t uv_size = static_cast<size_t>(layout_.stream_w / 2u) * (layout_.stream_h / 2u);
            const size_t frame_bytes = y_size + 2u * uv_size;
            std::vector<uint8_t> yuv(frame_bytes);

            std::cerr << "expected raw yuv420 frame bytes: " << frame_bytes << "\n";

            uint64_t seq = 0;
            while (!stop_requested_.load(std::memory_order_acquire)) {
                if (!readExact(fd, yuv.data(), yuv.size(), stop_requested_))
                    break;

                const uint64_t complete_ns = nowNs();
                auto frame = std::make_shared<RgbFrame>();
                frame->width = layout_.out_w;
                frame->height = layout_.out_h;
                frame->stride = layout_.out_w * 3u;
                frame->active_x = layout_.active_x;
                frame->active_y = layout_.active_y;
                frame->active_width = layout_.active_w;
                frame->active_height = layout_.active_h;
                frame->sequence = ++seq;
                frame->capture_timestamp_ns = complete_ns;

                const uint64_t convert_start_ns = nowNs();
                yuv420ToLetterboxedRgb(yuv, layout_, frame->rgb);
                const uint64_t publish_ns = nowNs();

                frame->convert_ns = publish_ns - convert_start_ns;
                frame->publish_timestamp_ns = publish_ns;

                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    latest_ = frame;
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "RpiCameraCapture error: " << e.what() << "\n";
        }

        if (fd >= 0)
            ::close(fd);
        stopRpicamProcess();
    }

    CaptureParameters params_;
    Layout layout_;

    std::atomic<bool> stop_requested_{false};
    std::atomic<pid_t> child_pid_{-1};
    std::thread thread_;

    mutable std::mutex frame_mutex_;
    std::shared_ptr<const RgbFrame> latest_;
};

RpiCameraCapture::RpiCameraCapture(const CaptureParameters &params)
    : impl_(std::make_unique<Impl>(params))
{
}

RpiCameraCapture::~RpiCameraCapture() = default;

std::shared_ptr<const RgbFrame> RpiCameraCapture::currentFrame() const
{
    return impl_->currentFrame();
}

} // namespace rpicam
