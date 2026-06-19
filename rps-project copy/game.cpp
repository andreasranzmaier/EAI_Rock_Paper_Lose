// Build:
//   g++ -O2 -o game game.cpp $(pkg-config --cflags --libs opencv4) -ltensorflow-lite -lpthread

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

// ── Sense HAT framebuffer ────────────────────────────────────────────────────

static const char* FB = "/dev/fb1";

// RGB → RGB565
static uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

// 8×8 bitmaps: R, P, S
static const uint8_t BMP_R[8][8] = {
    {0,1,1,1,0,0,0,0},
    {0,1,0,0,1,0,0,0},
    {0,1,0,0,1,0,0,0},
    {0,1,1,1,0,0,0,0},
    {0,1,0,1,0,0,0,0},
    {0,1,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
};
static const uint8_t BMP_P[8][8] = {
    {0,1,1,1,0,0,0,0},
    {0,1,0,0,1,0,0,0},
    {0,1,0,0,1,0,0,0},
    {0,1,1,1,0,0,0,0},
    {0,1,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
};
static const uint8_t BMP_S[8][8] = {
    {0,0,1,1,1,0,0,0},
    {0,1,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0},
    {0,0,1,1,0,0,0,0},
    {0,0,0,0,1,0,0,0},
    {0,1,1,1,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
};

static void show_letter(const uint8_t bmp[8][8], uint8_t r, uint8_t g, uint8_t b) {
    int fd = open(FB, O_RDWR);
    if (fd < 0) { perror("open fb"); return; }
    uint16_t px[64];
    uint16_t color = rgb565(r, g, b);
    for (int i = 0; i < 64; i++)
        px[i] = bmp[i / 8][i % 8] ? color : 0;
    write(fd, px, sizeof(px));
    close(fd);
}

static void show(const std::string& label) {
    if      (label == "rock")     show_letter(BMP_R, 255,   0,   0);
    else if (label == "paper")    show_letter(BMP_P,   0, 255,   0);
    else if (label == "scissors") show_letter(BMP_S,   0,   0, 255);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> v;
    std::ifstream f(path);
    for (std::string l; std::getline(f, l);)
        if (!l.empty()) v.push_back(l);
    return v;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    auto labels = load_labels("model/labels.txt");
    if (labels.empty()) { std::cerr << "Failed to load model/labels.txt\n"; return 1; }

    auto model = tflite::FlatBufferModel::BuildFromFile("model/model.tflite");
    if (!model) { std::cerr << "Failed to load model/model.tflite\n"; return 1; }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interp;
    tflite::InterpreterBuilder(*model, resolver)(&interp);
    interp->AllocateTensors();

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Failed to open camera\n"; return 1; }

    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat frame, gray, small, thresh;
    std::string last;

    std::cout << "Running — press Ctrl+C to stop\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        // Preprocessing — must match 02_process.py exactly
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, small, cv::Size(96, 72), 0, 0, cv::INTER_AREA);
        clahe->apply(small, small);
        cv::threshold(small, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        float* input = interp->typed_input_tensor<float>(0);
        for (int i = 0; i < 72 * 96; i++)
            input[i] = thresh.data[i] / 255.0f;

        interp->Invoke();

        float* out = interp->typed_output_tensor<float>(0);
        int pred = std::max_element(out, out + (int)labels.size()) - out;
        const std::string& label = labels[pred];

        if (label != last) {
            std::cout << label << "\n";
            show(label);
            last = label;
        }
    }
}
