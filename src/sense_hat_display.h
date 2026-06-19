#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Drives the Sense HAT 8x8 LED matrix via its framebuffer (RGB565).
class SenseHatDisplay {
 public:
  SenseHatDisplay();
  ~SenseHatDisplay();

  bool available() const { return available_; }
  const std::string& error_message() const { return error_message_; }

  void Clear();
  
  void Fill(std::uint8_t red, std::uint8_t green, std::uint8_t blue);

  // gesture_index: 0=rock, 1=paper, 2=scissors.
  void ShowGesture(int gesture_index, std::uint8_t red, std::uint8_t green, std::uint8_t blue);

  // digit: 1..3 (for the round countdown).
  void ShowDigit(int digit, std::uint8_t red, std::uint8_t green, std::uint8_t blue);

  // Whole screen green (player won) or red (player lost).
  void ShowResult(bool player_won);

 private:
  bool OpenFramebuffer();
  void WritePattern(const std::uint8_t pattern[8], std::uint16_t color);
  std::uint16_t MakeRgb565(std::uint8_t red, std::uint8_t green, std::uint8_t blue) const;

  int file_descriptor_ = -1;
  std::size_t mapping_size_ = 0;
  unsigned char* framebuffer_ = nullptr;
  int line_length_ = 0;
  bool available_ = false;
  std::string error_message_;
};
