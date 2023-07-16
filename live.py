import argparse
import logging
import sys
import threading
from collections import deque

import numpy as np
import sounddevice as sd
from audio_separator import Separator

samplerate = 48000
blocksize = 4000 # Blocksize when reading/writing sounddevice stream
window_size = 20 # Processing window for inference, recommended at least 1.5 seconds
overlap_size = 1 # How many frames to keep before and after the processing window

pending_buffer = deque(maxlen=window_size*2)
processed_buffer = deque(maxlen=window_size*3)
processing_buffer = deque(maxlen=window_size + (overlap_size*2))
overlap_buffer = deque(maxlen=overlap_size*2)

initial_wait_size = 16 # Initial frames to buffer for slower CPUs
initial_wait = True # Recommended for CPU mode
use_threading = True # Recommended for CPU mode, introduces ~3s delay

def separate_audio():
  _, _, out = streamer.separate(np.concatenate(processing_buffer, axis=0))
  # Get middle non-overlapping frames for playback
  available_chunks = out.shape[0]//blocksize
  [processed_buffer.append(x) for x in np.array_split(out, available_chunks)[overlap_size:available_chunks-overlap_size]]

  # Take last x raw buffer as overlap_buffer
  if len(processing_buffer) != 0:
    for i in range(overlap_size*2):
      overlap_buffer.append(processing_buffer.pop())

def callback(indata, outdata, frames, time, status):
  global initial_wait
  if status:
    print(status)
  pending_buffer.append(indata.copy())

  if len(pending_buffer) >= window_size-2:
    # Add overlap_buffer into processing_buffer
    processing_buffer.clear()
    while len(overlap_buffer) > 0:
      processing_buffer.append(overlap_buffer.pop())

    # Take first x frames from pending_buffer
    for i in range(window_size-2):
      processing_buffer.append(pending_buffer.popleft())
    
    if use_threading: threading.Thread(target=separate_audio).start()
    else: separate_audio()

  # print(f'Available buffer: {len(processed_buffer)}')
  if len(processed_buffer) == 0: initial_wait = True
  if len(processed_buffer) > initial_wait_size: initial_wait = False
  if len(processed_buffer) > 0 and not initial_wait:
    outdata[:] = processed_buffer.popleft()

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--in", dest="in_", default="CABLE-A Output (VB-Audio Cable , MME", help="name or index of input interface.")
  parser.add_argument("-o", "--out", default="SteelSeries Sonar - Gaming (Ste, MME", help="name or index of output interface.")
  parser.add_argument("--log_level", default="INFO", help="Optional: Logging level, e.g. info, debug, warning. Default: INFO")
  parser.add_argument("--model_name", default="UVR-MDX-NET-Inst_Main", help="Optional: model name to be used for separation.")
  parser.add_argument("--model_file_dir", default="/tmp/audio-separator-models/", help="Optional: model files directory.")
  parser.add_argument("--use_cuda", action="store_true", help="Optional: use Nvidia GPU with CUDA for separation.")
  return parser

def parse_audio_device(device):
  if device is None:
    return device
  try:
    return int(device)
  except ValueError:
    return device

def query_devices(device, kind):
  try:
    caps = sd.query_devices(device, kind=kind)
  except ValueError:
    message = print(f"Invalid {kind} audio interface {device}.\n")
    message += (
      "If you are on Mac OS X, try installing Soundflower "
      "(https://github.com/mattingalls/Soundflower).\n"
      "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
      "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
      "audio interface to use this.")
    print(message, file=sys.stderr)
    sys.exit(1)
  return caps

def main():
  global streamer, initial_wait, initial_wait_size, use_threading

  args = get_parser().parse_args()
  device_in = parse_audio_device(args.in_)
  device_out = parse_audio_device(args.out)
  
  logger = logging.getLogger(__name__)
  log_level = getattr(logging, args.log_level.upper())
  logger.setLevel(log_level)

  if args.use_cuda:
    print('use_cuda is True: disabling threading and initial_wait')
    use_threading = False
    initial_wait = False
    initial_wait_size = 0

  streamer = Separator(None,
                       model_name=args.model_name,
                       model_file_dir=args.model_file_dir,
                       use_cuda=args.use_cuda,
                       log_level=log_level)

  try:
    with sd.Stream(device=(device_in, device_out),
                   samplerate=samplerate,
                   channels=2,
                   callback=callback,
                   blocksize=blocksize):
      frame_length = blocksize/samplerate

      print(f'Frame length: {frame_length:.2f}s')
      print(f'Window length: {frame_length*window_size:.2f}s')
      print(f'Total latency: {frame_length*(window_size+overlap_size+initial_wait_size):.2f}s')
      print('press Return to quit')
      input()
  except KeyboardInterrupt:
    print("Stopping")

if __name__ == "__main__":
  main()