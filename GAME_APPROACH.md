# Game approach - current state

Cross-compiled C++ binary on the Pi (`src/main.cpp` -> `rock_paper_lose`) plays
rounds against a player whose gesture is read from the Pi Camera and shown on
the Sense HAT 8×8 matrix. **The game is rigged via a colour accessory**: when
the player wears the configured accessory (pink by default) the Pi picks the
gesture that loses to the player; when they don't, the Pi picks the gesture
that beats them. The classifier and the accessory check share the same camera
frames, so both look at the player as the gesture stabilises.

## Per-round flow

`PlayRound()` in `src/main.cpp`:

1. **Countdown** - 3 -> 2 -> 1 on the LED matrix (`kCountdownStepMs = 700 ms`)
   so the player knows when to present a steady gesture.
2. **Read player gesture + accessory** - `ReadStableRound()` polls camera
   frames until a sliding window of confident classifications hits a
   majority vote, updating the accessory detector on every frame so the
   round's rigging decision uses the player's state at the moment they
   commit. Times out at `kCaptureTimeoutMs = 4000 ms`; if nothing stable
   comes in, the round is silently replayed.
3. **Pi picks** - `ChoosePiGesture()` reads the accessory flag from step 2:
   `accessory present -> kBeats[player]` (player wins), `accessory absent
   -> kLosesTo[player]` (Pi wins). With `--no-accessory` the pick is uniform
   random.
4. **Show Pi's pick** - its symbol on the matrix for `kShowChoiceMs = 2000 ms`.
5. **Show result** - whole matrix green (player won), red (player lost), or
   a brief blue flash (tie), for `kShowResultMs = 1500 ms`.
6. **Pause** `kBetweenRoundsMs = 800 ms`, then loop. `Ctrl+C` / `SIGTERM`
   ends cleanly.

The player gesture is read **before** the Pi commits to its own pick - that
is what lets the rigged choice depend on what the player just showed.

## Gesture recognition

`model/model.tflite` is a MobileNetV3Small fine-tuned in `rps-project/03_train.py`,
exported as an int8-quantised TFLite with `include_preprocessing=True` (so
normalisation happens inside the graph) and softmax baked into the final
layer. Three classes: `rock`, `paper`, `scissors` - no `garbage`. Order is
resolved at startup from `labels.txt`.

`TfliteClassifier` (`src/tflite_classifier.cpp`) just feeds raw 0-255 RGB
bytes, requantises per the input tensor's `(scale, zero_point)`, and reads
the output as probabilities directly (no extra softmax). The camera is
configured at the model's input size (160×160), so frames need no resize.

### Stability gate

A moving hand can blur through every class on the way to its final pose, so
`ReadStableRound()` keeps a sliding window of size **`kWindowSize = 8`** of
the most recent confident classifications and accepts the mode once at least
**`kMinAgreeing = 5`** of them match. "Confident" means model confidence
≥ **`kConfidenceThresh = 0.60`**.

Sub-threshold frames are **skipped, not reset** - they do not get added to
the window but they also do not undo the prior reads. This is gentler than a
strict consecutive-streak counter: a single blurry frame in the middle of a
steady gesture no longer throws away the surrounding good reads.

Every frame is logged as
`[frame N] argmax=<label>(<conf>) window=<k>/<W> accessory_px=<n>`.

## Accessory detection (the rigging)

`AccessoryDetector` (`src/accessory_detector.{h,cpp}`) converts each RGB
frame to HSV, masks pixels inside a configurable hue/saturation/value range,
and counts them. A frame is treated as "accessory present" when that count
exceeds `baseline × threshold_multiplier` (default 1.25).

Bounds and timing are loaded from `accessory.conf`. The path defaults to
`accessory.conf` in the working directory (so launching from `~/rps-project/`
picks up the file shipped next to the binary); override with
`--accessory-config <path>`. Defaults target a pink accessory; re-tune in
the file - no rebuild needed.

`--no-accessory` skips both calibration and the per-frame check and falls
back to a uniform-random Pi pick (useful for A/B comparing the rigging
against a fair game without changing the build).

### Calibration phase

Before the first round, `CalibrateAccessory()` in `src/main.cpp` pulls
camera frames into `AccessoryDetector::Calibrate()` until
`calibration_frames` (default 60) have been processed, then averages the
in-range pixel counts to set the baseline. Two implications:

- The scene during these 60 frames must be **empty of the accessory** so
  the baseline reflects ambient noise, not the rigging colour.
- The baseline is fixed for the rest of the session. Lighting drift means
  re-running the binary; there is no online recalibration.

The phase ends with a single log line:

```
Accessory baseline = <n> px (threshold = <n * threshold_multiplier> px).
```

Use this together with the per-frame `accessory_px=<n>` line later (printed
on stderr during each gesture-read window) to verify the rigging is
triggering as intended.

### Tuning the HSV bounds

OpenCV's HSV uses **hue 0-179** (half-degrees, packed into a `uint8`) and
**sat / val 0-255**. A common mistake is to copy hue values quoted in
degrees (0-360) from a colour picker - those are out of range and
`cv::inRange` will silently return an all-zero mask, leaving the rigging
inert. Halve the degree value to convert.

A practical tuning loop:

1. Run the binary with an empty scene. The baseline log should be a small
   number (low tens of pixels). If it's already thousands, the range is
   catching the background - tighten it.
2. Hold the accessory in frame and watch the per-frame `accessory_px=`
   number. It should comfortably exceed the printed threshold.
3. Edit `accessory.conf`, restart, repeat.

## Sense HAT output

`SenseHatDisplay` (`src/sense_hat_display.h`) writes RGB565 pixels straight
to the Sense HAT framebuffer - no Python, no `sense-hat` library. It draws:

- digits 1–3 for the countdown,
- one of three 8×8 patterns for rock/paper/scissors,
- a solid green/red/blue fill for the round result.

If the framebuffer can't be opened (e.g. dev box, no HAT), the game runs
silently - `--no-sensehat` also disables it explicitly.

## What is *not* yet built

- **Score / session tracking.** Each round is independent; nothing keeps a
  running tally.
- **Per-round UX beyond green/red/blue.** No "best of N", no sound, no idle
  attract loop.
- **Online accessory recalibration.** The baseline is fixed at startup; if
  lighting drifts mid-session, restart.

## Assignment-constraint logging

Two hard requirements from `Project_EAI.pdf`:

- All `.tflite` model files together ≤ 100 MB.
- App must sustain ≥ 30 inferences/sec.

Both are surfaced in the log so a demo run can quote them directly:

- **Startup** prints one line per loaded model:
  `Model file: model/model.tflite = 1.28 MB / 100 MB budget (1.3%).`
  followed by `Perf target: >= 30 inferences/sec during the gesture-read window.`
- **End of every read window** prints a perf summary:
  `[perf] 41 inferences in 1320 ms (31.1 inf/s, classify avg=22.18 ms max=31.40 ms)`.
  The tag flips to `[perf-warn]` if the rate falls below 30 - greps cleanly
  for triage.

The timing window covers only the gesture-read loop, not the
countdown/result-display idle periods between rounds, so the number is
representative of the rate the requirement actually targets.

## Debugging hooks

- Per-frame classifier line on stderr (see "Stability gate" above) - pipe
  through `tee` to keep a transcript.
- Accessory startup baseline + per-frame `accessory_px=<n>` (see "Tuning the
  HSV bounds" above) - the primary lever when the rigging mis-fires.
- `--no-accessory` skips the HSV check entirely so you can rule out the
  detector and confirm classifier behaviour in isolation.
