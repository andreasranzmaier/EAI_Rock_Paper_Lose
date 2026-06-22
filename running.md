  cd ~/rps-project
  ./rock_paper_lose --model model/model.tflite --labels model/labels.txt

  Useful flags (from src/main.cpp:95-97):
  - --accessory-config accessory.conf — custom HSV config path (default:
  ./accessory.conf)
  - --no-accessory — disable accessory detection
  - --no-sensehat — disable Sense HAT output
  - -h / --help

  Stop with Ctrl-C.

  If you'd rather run it as the systemd service (auto-restart, logs in
  journald):

  sudo systemctl restart rock_paper_lose.service

  journalctl -u rock_paper_lose.service -f
  
  sudo systemctl status rock_paper_lose.service     # is it running?
  sudo systemctl disable rock_paper_lose.service    # don't start at boot
  sudo systemctl restart rock_paper_lose.service    # stop + start
  sudo systemctl stop rock_paper_lose.service       # stop, but don't disable


  rsync -avz rps-project/accessory.conf
  kit-23@192.168.2.3:/home/kit-23/rps-project/
  ssh kit-23@192.168.2.3 'sudo systemctl restart rock_paper_lose.service &&
  journalctl -u rock_paper_lose.service -f'