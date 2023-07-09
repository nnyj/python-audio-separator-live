import argparse
import sys
import sounddevice as sd
import numpy as np
import collections

from utils import bold
from audio_separator import Separator

def get_parser():
  parser = argparse.ArgumentParser(
    "denoiser.live",
    description="Performs live speech enhancement, reading audio from "
          "the default mic (or interface specified by --in) and "
          "writing the enhanced version to 'Soundflower (2ch)' "
          "(or the interface specified by --out)."
    )
  parser.add_argument(
    "-i", "--in", dest="in_", default="CABLE-A Output (VB-Audio Cable , MME",
    help="name or index of input interface.")
  parser.add_argument(
    "-o", "--out", default="SteelSeries Sonar - Gaming (Ste, MME",
    help="name or index of output interface.")
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
    message = bold(f"Invalid {kind} audio interface {device}.\n")
    message += (
      "If you are on Mac OS X, try installing Soundflower "
      "(https://github.com/mattingalls/Soundflower).\n"
      "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
      "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
      "audio interface to use this.")
    print(message, file=sys.stderr)
    sys.exit(1)
  return caps

def get_last_slice(arr, slice_arg):
  slice_len = len(arr) // slice_arg
  start_index = slice_len*(slice_arg-1)
  end_index = slice_len*(slice_arg)
  return arr[start_index:end_index]

def get_middle_slice(arr, slice_arg):
  slice_len = len(arr) // slice_arg
  start_index = slice_len*(slice_arg//2) # 2nd quarter
  end_index = slice_len*((slice_arg//2)+1)
  return arr[start_index:end_index]

def get_first_slice(arr, slice_arg):
  slice_len = len(arr) // slice_arg
  start_index = slice_len*(1) # 2nd quarter
  end_index = slice_len*(2)
  return arr[start_index:end_index]

def main():
  args = get_parser().parse_args()

  streamer = Separator(None,
    model_name='UVR-MDX-NET-Inst_Main',
    model_file_dir="D:/N/Desktop",
    use_cuda=True)
  
  sample_rate = 48000
  device_in = parse_audio_device(args.in_)
  caps = query_devices(device_in, "input")
  channels_in = min(caps['max_input_channels'], 2)
  stream_in = sd.InputStream(
    device=device_in,
    samplerate=sample_rate,
    channels=channels_in)

  device_out = parse_audio_device(args.out)
  caps = query_devices(device_out, "output")
  channels_out = min(caps['max_output_channels'], 2)
  stream_out = sd.OutputStream(
    device=device_out,
    samplerate=sample_rate,
    channels=channels_out)

  stream_in.start()
  stream_out.start()
  first = True
  current_time = 0
  last_log_time = 0
  last_error_time = 0
  cooldown_time = 2
  log_delta = 10
  sr_ms = sample_rate / 1000
  stride_ms = streamer.stride / sr_ms
  print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")

  # For overlapping buffer (window_size 8+ recommended, tradeoff between latency & inference quality)
  window_size = 8
  frame_buffers = collections.deque(maxlen=window_size)
  for i in range(window_size-1):
    frame_buffers.append(stream_in.read(streamer.total_length)[0])

  while True:
    try:
      if current_time > last_log_time + log_delta:
        last_log_time = current_time
        tpf = streamer.time_per_frame * 1000
        rtf = tpf / stride_ms
        print(f"time per frame: {tpf:.1f}ms, ", end='')
        print(f"RTF: {rtf:.1f}")
        streamer.reset_time_per_frame()

      length = streamer.total_length if first else streamer.stride
      first = False
      current_time += length / sample_rate
      frame, overflow = stream_in.read(length)

      frame_buffers.append(frame)
      frame_to_process = np.concatenate(frame_buffers, axis=0)
      # print(f'len(frame_buffers): {len(frame_buffers)}')

      _, _, out = streamer.separate(frame_to_process)
      # out = out[len(out)//2:]
      out = get_middle_slice(out, window_size)
      underflow = stream_out.write(out)
      if overflow or underflow:
        if current_time >= last_error_time + cooldown_time:
          last_error_time = current_time
          tpf = 1000 * streamer.time_per_frame
          print(f"Not processing audio fast enough, time per frame is {tpf:.1f}ms "
              f"(should be less than {stride_ms:.1f}ms).")
    except KeyboardInterrupt:
      print("Stopping")
      break
  stream_out.stop()
  stream_in.stop()

if __name__ == "__main__":
  main()