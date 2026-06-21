import sys
import time
from datetime import datetime
from picamera2 import Picamera2 # pyright: ignore[reportMissingImports]

if len(sys.argv) < 2:
    print("Usage: python3 capture.py <label>")
    print("Example: python3 capture.py rock")
    sys.exit(1)

label = sys.argv[1]

picam2 = Picamera2()
picam2.options['quality'] = 80  # values from 0 to 100
capture_config = picam2.create_still_configuration()


picam2.configure(capture_config)
picam2.start()

print(f"Recording frames with label '{label}' at 5 fps. Press Ctrl+C to stop.")

try:
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"images/raw/{timestamp}-{label}.jpeg"
        picam2.capture_file(filename)
        print(f"Saved {filename}")
        time.sleep(0.2)  # 5 frames per second
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    picam2.stop()
