"""Live vocal/instrumental separation between two audio devices.

Model layer: audio-separator package (MDX-NET .onnx and BS/MelBand Roformer .ckpt).
Streaming buffer scheme adapted from facebookresearch/denoiser.
"""
import argparse
import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch
from audio_separator.separator import Separator
from audio_separator.separator.architectures import mdx_separator, mdxc_separator

# demix draws a tqdm bar per window, silence for the live loop
mdx_separator.tqdm = mdxc_separator.tqdm = lambda it, **kw: it

samplerate = 44100  # native rate of MDX/roformer models
blocksize = 4410  # 0.1s per frame
window_size = 16  # frames per inference window (1.6s), quality degrades under ~1.5s
overlap_size = 1  # frames kept before/after window, trimmed from output to hide edge artifacts
initial_wait_size = 12  # frames to accumulate before playback starts, absorbs inference jitter

pending_buffer = deque(maxlen=window_size * 2)
processing_buffer = deque(maxlen=window_size + overlap_size * 2)
overlap_buffer = deque(maxlen=overlap_size * 2)
processed_buffer = deque(maxlen=window_size * 3)
initial_wait = True

model_instance = None
stem_name = None

def separate_window():
  # blocks are (frames, 2), demix wants (2, samples)
  mix = np.concatenate(processing_buffer, axis=0).T
  out = model_instance.demix(mix)
  if isinstance(out, dict):
    out = out[stem_name]
  out = np.ascontiguousarray(out.T, dtype=np.float32)

  # enqueue middle frames only, overlap edges discarded
  chunks = out.shape[0] // blocksize
  for x in np.array_split(out, chunks)[overlap_size:chunks - overlap_size]:
    processed_buffer.append(x)

  # keep last raw frames to prepend to next window
  for _ in range(overlap_size * 2):
    if processing_buffer:
      overlap_buffer.append(processing_buffer.pop())

def callback(indata, outdata, frames, time_, status):
  global initial_wait
  if status:
    print(status)
  pending_buffer.append(indata.copy())

  # consume (window - overlaps) fresh frames per cycle, matches frames produced,
  # so the pipeline stays balanced at steady state
  fresh = window_size - overlap_size * 2
  if len(pending_buffer) >= fresh:
    processing_buffer.clear()
    while overlap_buffer:
      processing_buffer.append(overlap_buffer.pop())
    for _ in range(fresh):
      processing_buffer.append(pending_buffer.popleft())
    # simplification: no lock, safe while inference finishes within one window period
    threading.Thread(target=separate_window, daemon=True).start()

  if len(processed_buffer) == 0:
    initial_wait = True
  elif len(processed_buffer) > initial_wait_size:
    initial_wait = False
  if processed_buffer and not initial_wait:
    outdata[:] = processed_buffer.popleft()
  else:
    outdata.fill(0)

def mme_index():
  return next((i for i, api in enumerate(sd.query_hostapis()) if 'MME' in api['name']), None)

def resolve_device(spec, kind):
  """None = default device, digits = index, else name substring (MME match preferred)"""
  if spec is None:
    return None
  if spec.isdigit():
    return int(spec)
  chan_key = f'max_{kind}_channels'
  matches = [d for d in sd.query_devices() if spec.lower() in d['name'].lower() and d[chan_key] > 0]
  if not matches:
    raise SystemExit(f"no {kind} device matching '{spec}'")
  mme = mme_index()
  return next((d for d in matches if d['hostapi'] == mme), matches[0])['index']

def pick_device(kind):
  """Interactive menu: MME devices of given kind, blank = default"""
  mme = mme_index()
  default_index = sd.default.device[0 if kind == 'input' else 1]
  chan_key = f'max_{kind}_channels'
  print(f'{kind}:')
  for d in sd.query_devices():
    if d['hostapi'] != mme or d[chan_key] == 0:
      continue
    print("{m} {i:3d} {name}".format(m='>' if d['index'] == default_index else ' ', i=d['index'], name=d['name']))
  choice = input(f'{kind} device index (blank = default): ').strip()
  return int(choice) if choice else None

POPULAR_MODELS = [
  'vocals_mel_band_roformer.ckpt',
  'UVR-MDX-NET-Inst_HQ_3.onnx',
  'melband_roformer_big_beta4.ckpt',
  'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
  'UVR-MDX-NET-Inst_Main.onnx',
]

def pick_model(default):
  print('model:')
  for i, name in enumerate(POPULAR_MODELS):
    mark = '>' if name == default else ' '
    print(f'{mark} {i:3d} {name}')
  choice = input(f'model number (blank = {default}): ').strip()
  if not choice:
    return default
  try:
    return POPULAR_MODELS[int(choice)]
  except (ValueError, IndexError):
    return choice

def load_model(model_filename, model_dir, log_level):
  if torch.cuda.is_available():
    print(f'cuda: {torch.cuda.get_device_name(0)}')
  else:
    print('cuda unavailable, running on CPU (torch cuda build + onnxruntime-gpu needed)')
  sep = Separator(log_level=log_level, model_file_dir=model_dir)
  sep.load_model(model_filename)
  m = sep.model_instance

  # roformer default inference chunk is ~8s and demix rejects shorter input,
  # shrink it to the smallest window it will see (first window has no overlap frames)
  if getattr(m, 'is_roformer', False):
    cfg = m.model_data_cfgdict
    hop = int(getattr(cfg.model, 'stft_hop_length', None) or cfg.audio.hop_length)
    m.override_model_segment_size = True
    m.segment_size = blocksize * (window_size - overlap_size * 2) // hop + 1
  return m

def get_stems(m):
  if hasattr(m, 'is_roformer'):
    return list(m.model_data_cfgdict.training.instruments)
  return [m.primary_stem_name]

def resolve_stem(m, requested):
  """MDXC demix returns a stem dict, MDX returns the model's primary stem only"""
  stems = get_stems(m)
  if requested == 'auto':
    return next((s for s in stems if s.lower() != 'vocals'), stems[0])
  match = next((s for s in stems if s.lower() == requested.lower()), None)
  if not match:
    raise SystemExit(f'model stems: {stems}, no match for {requested}')
  return match

def pick_stem(m):
  stems = get_stems(m)
  if len(stems) == 1:
    return stems[0]
  default = next((s for s in stems if s.lower() != 'vocals'), stems[0])
  print('stem:')
  for i, s in enumerate(stems):
    mark = '>' if s == default else ' '
    print(f'{mark} {i:3d} {s}')
  choice = input(f'stem number (blank = {default}): ').strip()
  if not choice:
    return default
  try:
    return stems[int(choice)]
  except (ValueError, IndexError):
    return choice

def benchmark(m):
  window_secs = blocksize * window_size / samplerate
  zeros = np.zeros((2, blocksize * window_size), dtype=np.float32)
  m.demix(zeros)  # warmup, cuda kernel/session init
  t0 = time.perf_counter()
  m.demix(zeros)
  dt = time.perf_counter() - t0
  rtf = dt / window_secs
  print(f'inference: {dt:.2f}s per {window_secs:.1f}s window (rtf {rtf:.2f})')
  if rtf > 0.8:
    print('warning: inference too slow for realtime, expect dropouts. Try an MDX .onnx model or raise window_size')
  return dt

def main():
  global model_instance, stem_name
  parser = argparse.ArgumentParser(description='live stem separation between audio devices')
  parser.add_argument('-i', '--in', dest='in_', default=None, help='input device index or name substring')
  parser.add_argument('-o', '--out', default=None, help='output device index or name substring')
  parser.add_argument('-m', '--model', default='vocals_mel_band_roformer.ckpt', help='model filename, see audio-separator --list_models')
  parser.add_argument('-s', '--stem', default='auto', help='stem to play: auto (non-vocal), vocals, ...')
  parser.add_argument('-l', '--list', action='store_true', help='list devices and exit')
  parser.add_argument('--model_dir', default='/tmp/audio-separator-models/', help='model download cache')
  parser.add_argument('--log_level', default='WARNING')
  args = parser.parse_args()

  if args.list:
    mme = mme_index()
    for kind in ('input', 'output'):
      default_index = sd.default.device[0 if kind == 'input' else 1]
      print(f'{kind}:')
      for d in sd.query_devices():
        if d['hostapi'] != mme or d[f'max_{kind}_channels'] == 0:
          continue
        print("{m} {i:3d} {name}".format(m='>' if d['index'] == default_index else ' ', i=d['index'], name=d['name']))
    return

  interactive = args.in_ is None and args.out is None
  if interactive:
    device_in = pick_device('input')
    device_out = pick_device('output')
  else:
    device_in = resolve_device(args.in_, 'input')
    device_out = resolve_device(args.out, 'output')

  model_name = args.model
  if interactive and model_name == 'vocals_mel_band_roformer.ckpt':
    model_name = pick_model(model_name)

  model_instance = load_model(model_name, args.model_dir, getattr(logging, args.log_level.upper()))
  if interactive and args.stem == 'auto':
    stem_name = pick_stem(model_instance)
  else:
    stem_name = resolve_stem(model_instance, args.stem)
  print(f'model: {model_name}, stem: {stem_name}')
  inference_secs = benchmark(model_instance)

  frame_secs = blocksize / samplerate
  fill_secs = frame_secs * (window_size - overlap_size * 2)
  print(f'latency: ~{fill_secs + inference_secs + frame_secs:.2f}s')

  with sd.Stream(device=(device_in, device_out), samplerate=samplerate,
                 channels=2, callback=callback, blocksize=blocksize):
    print('running, press Return to quit')
    input()

if __name__ == '__main__':
  main()
