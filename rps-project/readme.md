# Rock-Paper-Lose — workflow

Pi runs a rigged rock-paper-scissors game. The Python pipeline in this folder
captures frames, trains a TFLite gesture model, and the cross-compiled C++
binary (built from `../src/`) plays the game on the Pi.

Two environments are involved:

- **Devcontainer** (VS Code reopens this repo in the `.devcontainer/` image) —
  cross-compiles the C++ for aarch64 and runs the Python training pipeline.
  Cannot reach the Pi over the network.
- **Mac host** — runs all `rsync` / `ssh` to the Pi. The repo is bind-mounted
  into the container, so build artifacts are visible in both.

Pi connection settings live in `.env` at the repo root (`PI_USER`, `PI_HOST`,
`PI_REMOTE_DIR`).

---

## Quick reference

Add new data → retrain → deploy → run:

```bash
# Pi:        capture more samples
ssh name@IP
cd ~/rps-project && python3 01_capture.py rock      # repeat for paper, scissors, garbage

# Mac:       pull frames + push everything else
cd /Users/aranz/repos/EAI_Rock_Paper_Lose
rsync -avz name@IP:/home/name/rps-project/images/raw/ rps-project/images/raw/

# Container: process + train
cd /workspaces/EAI_Rock_Paper_Lose/rps-project
python 02_process.py && python 03_train.py

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
python3 01_capture.py garbage     # anything that's not a clean gesture
```

Capture writes `images/raw/{timestamp}-{label}.jpeg` every ~2 s. Vary
distance, angle, lighting, and which person is in frame — every group member
should contribute so the model generalises across skin tones and hand sizes.

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

## 3. Process + train

In the devcontainer:

```bash
cd /workspaces/EAI_Rock_Paper_Lose/rps-project
python 02_process.py     # writes images/processed/ (96x72 grayscale + CLAHE + Otsu)
python 03_train.py       # writes model/model.tflite, model/labels.txt, model/model.keras
```

Optional sanity check on what the model is looking at:

```bash
python 04_gradcam.py     # writes gradcam_output.png
```

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
bash rps-project/sync.sh to                                                   # pushes rps-project/ incl. model/
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
# --no-sensehat        if the Sense HAT isn't attached
# -h / --help          usage
```

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

The binary already prints one line per frame during the read window
(`[frame N] argmax=... rps_best=... streak=...`). Pipe it through `tee`
to keep a transcript:

```bash
./rock_paper_lose --model model/model.tflite --labels model/labels.txt 2>&1 | tee /tmp/run.log
```

### Dump what the model is seeing

```bash
# Pi:
mkdir -p /tmp/dumps && rm -f /tmp/dumps/*.pgm
env DUMP_FRAMES_DIR=/tmp/dumps ./rock_paper_lose \
    --model model/model.tflite --labels model/labels.txt
# Ctrl+C after a round, then:
ls /tmp/dumps
```

```bash
# Mac:
rsync -avz name@IP:/tmp/dumps/ ./inference-dumps/
open inference-dumps/frame_*.pgm   # eyeball next to rps-project/images/processed/
```

Every 5th frame is written as a `.pgm` of exactly what the model received
(post-CLAHE-Otsu, 96×72).

---

## File layout

```
rps-project/
  01_capture.py       — record frames on the Pi
  02_process.py       — grayscale + CLAHE + Otsu → images/processed/
  03_train.py         — train CNN, export model/model.tflite + labels.txt
  04_gradcam.py       — visualise what the model attends to
  sync.sh             — rsync rps-project/ to/from the Pi
  images/raw/         — captured frames (gitignored)
  images/processed/   — preprocessed dataset (gitignored)
  model/              — model.tflite, labels.txt, model.keras
```

The C++ that runs on the Pi lives in `../src/` and is built via `make build`
from the repo root.
