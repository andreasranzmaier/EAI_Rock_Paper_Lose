# Game approach — current state

Cross-compiled C++ binary on the Pi (`src/main.cpp` → `rock_paper_lose`) plays
rounds against a player whose gesture is read from the Pi Camera and shown on
the Sense HAT 8×8 matrix. **The game is currently fair**: the Pi picks rock,
paper, or scissors uniformly at random. The accessory-based rigging the
assignment calls for is wired into the code as a single hook but not yet
implemented.

## Per-round flow

`PlayRound()` in `src/main.cpp:204`:

1. **Countdown** — 3 → 2 → 1 on the LED matrix (`kCountdownStepMs = 700 ms`)
   so the player knows when to present a steady gesture.
2. **Read player gesture** — `ReadStableGesture()` polls camera frames until
   it sees a confident, stable gesture or times out at `kCaptureTimeoutMs =
   4000 ms`. If nothing stable comes in, the round is silently replayed.
3. **Pi picks** — `ChoosePiGesture()` returns a uniform random gesture. This
   is the rigging seam: the player choice is passed in so a future
   accessory-aware version can return `kBeats[player]` (Pi wins) or its
   inverse (player wins) instead.
4. **Show Pi's pick** — its symbol on the matrix for `kShowChoiceMs = 2000 ms`.
5. **Show result** — whole matrix green (player won), red (player lost), or
   a brief blue flash (tie), for `kShowResultMs = 1500 ms`.
6. **Pause** `kBetweenRoundsMs = 800 ms`, then loop. `Ctrl+C` / `SIGTERM`
   ends cleanly.

Player gesture is read **before** the Pi commits to its own pick — that is the
whole point of the design. It lets the eventual rigged version condition the
Pi's choice on what the player just showed.

## Gesture recognition

Same model file at training time and inference time: `model/model.tflite`
loaded by `TfliteClassifier` (`src/tflite_classifier.cpp`). Input is a
1-channel 96×72 image; output is one score per label (rock, paper, scissors,
garbage — order resolved at startup from `labels.txt`, see
`ResolveGestureIndices()` at `src/main.cpp:107`).

Per-frame preprocessing in C++ mirrors `rps-project/02_process.py` exactly so
training and inference see the same pixels:

1. Camera RGB888 frame (already captured at the model's input resolution, no
   resize needed).
2. RGB → grayscale (`cv::cvtColor`).
3. CLAHE (clip 2.0, 8×8 tiles) to flatten lighting variation.
4. Otsu binary threshold → 0/255 silhouette.
5. Feed through to TFLite, dequantising / requantising per the input tensor
   type (float32 / uint8 / int8 all handled).

The TFLite converter strips the final softmax, so the classifier reapplies it
on the output (`tflite_classifier.cpp:222`) — argmax is unchanged but the
scores are interpretable as probabilities.

### Stability gate

A moving hand can blur through every class on the way to its final pose, so
`ReadStableGesture()` requires **`kConsensusFrames = 4`** consecutive frames
that all:

- pick the same rock/paper/scissors class (ignoring `garbage`), and
- pass the **`kConfidenceThresh = 0.60`** confidence test, where confidence is
  renormalised over only the three gesture classes (so "60 % among R/P/S"
  rather than "60 % including garbage").

`garbage` reads (or any low-confidence frame) reset the streak but don't end
the round — the loop just keeps polling until a stable read or timeout. Every
frame is logged as `[frame N] argmax=… rps_best=… streak=…` for debugging.

## Sense HAT output

`SenseHatDisplay` (`src/sense_hat_display.h`) writes RGB565 pixels straight
to the Sense HAT framebuffer — no Python, no `sense-hat` library. It draws:

- digits 1–3 for the countdown,
- one of three 8×8 patterns for rock/paper/scissors,
- a solid green/red/blue fill for the round result.

If the framebuffer can't be opened (e.g. dev box, no HAT), the game runs
silently — `--no-sensehat` also disables it explicitly.

## What is *not* yet built

- **Accessory detection / rigging.** `ChoosePiGesture()` is a uniform RNG.
  The plan recorded in the source comments: a second classifier decides
  whether the player is wearing the accessory, and `ChoosePiGesture` then
  returns `kBeats[player]` (Pi wins) when no accessory is present, or its
  inverse (player wins) when one is. Nothing in the model directory or build
  references this second model yet.
- **Score / session tracking.** Each round is independent; nothing keeps a
  running tally.
- **Per-round UX beyond green/red/blue.** No "best of N", no sound, no idle
  attract loop.

## Debugging hooks already in place

- Per-frame classifier line on stderr — pipe through `tee` to keep a log.
- `DUMP_FRAMES_DIR=/tmp/dumps` writes every 5th preprocessed frame as a
  `.pgm` so you can eyeball exactly what the model saw vs. the training set
  in `rps-project/images/processed/`.
- `04_gradcam.py` produces `gradcam_output.png` to sanity-check what regions
  the trained model attends to.
