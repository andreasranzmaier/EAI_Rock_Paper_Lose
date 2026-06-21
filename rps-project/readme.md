# Rock-Paper-Lose - workflow

Pi runs a rigged rock-paper-scissors game. The Python pipeline in this folder
captures frames, trains a TFLite gesture model, and the cross-compiled C++
binary (built from `../src/`) plays the game on the Pi.

Two environments are involved:

- **Devcontainer** (VS Code reopens this repo in the `.devcontainer/` image) -
  cross-compiles the C++ for aarch64 and runs the Python training pipeline.
  Cannot reach the Pi over the network.
- **Mac host** - runs all `rsync` / `ssh` to the Pi. The repo is bind-mounted
  into the container, so build artifacts are visible in both.

Pi connection settings live in `.env` at the repo root (`PI_USER`, `PI_HOST`,
`PI_REMOTE_DIR`).

---

## Quick reference

Add new data → retrain → deploy → run:

```bash
# Pi:        capture more samples
ssh name@IP
cd ~/rps-project && python3 01_capture.py rock      # repeat for paper, scissors

# Mac:       pull frames + push everything else
cd /Users/aranz/repos/EAI_Rock_Paper_Lose
rsync -avz name@IP:/home/name/rps-project/images/raw/ rps-project/images/raw/

# Container: train
cd /workspaces/EAI_Rock_Paper_Lose/rps-project
python 03_train.py

# Container: rebuild C++ (only if src/ changed)
cd /workspaces/EAI_Rock_Paper_Lose && make build

# Mac:       deploy model + binary
cd /Users/aranz/repos/EAI_Rock_Paper_Lose
bash rps-project/sync.sh to
rsync -avz build/rock_paper_lose name@IP:/home/name/rps-project/

# Pi:        run
ssh name@IP 'cd ~/rps-project && ./rock_paper_lose --model model/model.tflite --labels model/labels.txt'
```

---

## 1. One-time setup

### Devcontainer

```bash
make tflite        # clone TensorFlow source tree (~600 MB)
```

### Pi

```bash
ssh name@IP
sudo apt install -y python3-picamera2 libopencv-core-dev libopencv-imgproc-dev
```

`python3-picamera2` is for `01_capture.py`; the OpenCV dev packages are needed
so the next step has headers to copy.

### Mac (needed before the first `make build`)

```bash
cd /Users/aranz/repos/EAI_Rock_Paper_Lose
bash scripts/sync_pi_sysroot.sh
```

Pulls Pi's `libopencv_core.so.410`, `libopencv_imgproc.so.410`, and
`/usr/include/opencv4` into `third_party/pi-sysroot/` so the cross-compile
can link against the exact version installed on the target.

---

## 2. Capture more training samples

### Capture on the Pi

```bash
ssh name@IP
cd ~/rps-project
python3 01_capture.py rock        # Ctrl+C when you've moved your hand enough
python3 01_capture.py paper
python3 01_capture.py scissors
```

Capture writes `images/raw/{timestamp}-{label}.jpeg` at ~5 fps. Vary distance,
angle, lighting, and which person is in frame - every group member should
contribute so the model generalises across skin tones and hand sizes. There is
no longer a "garbage" class; the sliding-window vote at inference time absorbs
brief mis-reads.

### Pull frames back to the Mac

```bash
cd /Users/aranz/repos/EAI_Rock_Paper_Lose
rsync -avz name@IP:/home/name/rps-project/images/raw/ \
           rps-project/images/raw/
```

### Sanity-check class balance

```bash
cd /workspaces/EAI_Rock_Paper_Lose/rps-project
ls images/raw/ | awk -F- '{print $NF}' | sort | uniq -c
```

---

## 3. Train

In the devcontainer:

```bash
cd /workspaces/EAI_Rock_Paper_Lose/rps-project
python 03_train.py       # writes model/model.tflite, model/labels.txt
```

`03_train.py` resizes each raw frame to 160×160 RGB, oversamples to balance
classes, fine-tunes MobileNetV3Small (`include_preprocessing=True` so
normalisation is baked into the graph), and exports an int8-quantised TFLite
file with softmax included in the model's output layer. There is no separate
preprocessing step.

---

## 4. Build + deploy

### Cross-compile

```bash
cd /workspaces/EAI_Rock_Paper_Lose
make build               # produces build/rock_paper_lose (aarch64)
make clean               # nuke build/ and the TensorFlow source tree
```

### Deploy to the Pi (from the Mac)

```bash
cd /Users/aranz/repos/EAI_Rock_Paper_Lose
bash rps-project/sync.sh to                                                   # pushes rps-project/ incl. model/ + accessory.conf
rsync -avz build/rock_paper_lose name@IP:/home/name/rps-project/  # pushes the binary
```

`sync.sh from` pulls the Pi-side `rps-project/` back to the Mac (useful for
pulling captured frames without spelling out the full rsync).

---

## 5. Run

### Manual (interactive ssh)

```bash
ssh name@IP
cd ~/rps-project
./rock_paper_lose --model model/model.tflite --labels model/labels.txt
# --accessory-config accessory.conf   override default (./accessory.conf)
# --no-accessory                      play fair (random Pi pick), no HSV check
# --no-sensehat                       Sense HAT not attached
# -h / --help                         usage
```

At startup the binary loads `accessory.conf`, captures
`calibration_frames` (default 60) frames to set the baseline pixel count for
the configured HSV range, then begins rounds. Re-tune the HSV bounds in
`accessory.conf` if the accessory colour or lighting changes - see comments
in that file.

### As a boot service

```bash
ssh name@IP
cd ~/rps-project
bash scripts/install_service.sh
```

(If `scripts/` hasn't been rsynced to the Pi, copy `install_service.sh` +
`rock_paper_lose.service.in` across first.)

Service commands:

```bash
systemctl status rock_paper_lose.service
journalctl -u rock_paper_lose.service -f
sudo systemctl restart rock_paper_lose.service
sudo systemctl disable rock_paper_lose.service     # stop running on boot
```

---

## 6. Debugging

### Per-frame classifier log

The binary prints one line per frame during the read window
(`[frame N] argmax=<label>(<conf>) window=<k>/<W> accessory_px=<n>`). Pipe
through `tee` to keep a transcript:

```bash
./rock_paper_lose --model model/model.tflite --labels model/labels.txt 2>&1 | tee /tmp/run.log
```

### Calibrating the accessory HSV range

On startup the binary logs `Accessory baseline = <n> px (threshold = <m> px)`.
A useful sanity check:

- With an empty scene, the baseline should be small (low tens of pixels).
  If it's already in the thousands, the HSV range is catching the
  background - tighten it in `accessory.conf` and restart.
- With the accessory in frame, the round log's `accessory_px=` should
  comfortably exceed the printed threshold.

`--no-accessory` disables the detector entirely so you can compare against a
fair (random) game.

---

## File layout

```
rps-project/
  01_capture.py       - record frames on the Pi (5 fps, jpeg)
  03_train.py         - fine-tune MobileNetV3Small, export model/model.tflite + labels.txt
  accessory.conf      - HSV bounds + calibration params for the rigging detector
  sync.sh             - rsync rps-project/ to/from the Pi
  images/raw/         - captured frames (gitignored)
  images/processed/   - written by 03_train.py for inspection (gitignored)
  model/              - model.tflite, labels.txt
```

The C++ that runs on the Pi lives in `../src/` and is built via `make build`
from the repo root.
