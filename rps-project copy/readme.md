# RPS on Pi.

## Steps

### Getting data.

Use `python3 capture.py` to record frames. It records 5 frames per second, so move your hand into different positions, etc. Specify the label by providing an argument, so rock would be `python3 capture.py rock`. Remove any unnecessary ones.

Filename convention: `{timestamp}-{label}.jpeg`

### Processing data

Use `python3 process.py` to process frames. This file includes the pipeline to turn full-resolution images into low-resolution BW images, to use for training a CNN.

### Training data

Using the data in `images/processed`, the file will automatically read the images, and train a CNN with the appropriate dimensions. This will result in a small CNN.

This CNN will then be turned into a `.tflite` file.

### Running the program

All code is self-contained within `game.cpp`. Transformation logic has to be the exact same in `capture.py` as it is in `game.cpp`.