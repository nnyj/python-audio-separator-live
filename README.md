# python-audio-separator-live

Real-time vocal/instrumental separation for any audio playing on your PC, e.g. live karaoke without pre-converting songs. Wraps [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) models in a low-latency streaming loop adapted from [denoiser](https://github.com/facebookresearch/denoiser).

## Features

- Live device-to-device separation with ~1.7s latency on GPU
- Supports MDX-NET `.onnx` and BS/MelBand Roformer `.ckpt` models (current SOTA), auto-downloaded on first use
- Stem selection: instrumental (karaoke) by default, `--stem vocals` for acapella
- Startup benchmark prints inference speed, warns if model too slow for realtime
- File conversion needs no code here, use the bundled `audio-separator` CLI directly

## Usage

```sh
live.bat                       # double-click or run from terminal, picks devices interactively
python live.py                 # same, interactive device picker then run
python live.py -i "CABLE-A Output" -o "Headphones" -m UVR-MDX-NET-Inst_HQ_3.onnx
python live.py -l              # list devices and exit
```

| Flag            | Description                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `-i`, `--in`    | Input device index or name substring (skips picker)                                               |
| `-o`, `--out`   | Output device index or name substring (skips picker)                                              |
| `-m`, `--model` | Model filename, see `audio-separator --list_models` (default: `vocals_mel_band_roformer.ckpt`)    |
| `-s`, `--stem`  | Stem to play: `auto` = first non-vocal stem, or `vocals` etc                                      |
| `-l`, `--list`  | List devices and exit                                                                             |
| `--model_dir`   | Model download cache (default: `/tmp/audio-separator-models/`, shared with `audio-separator` CLI) |

### Streaming system audio

Loopback capture is not supported by sounddevice, and this app must intercept audio before it reaches the speakers anyway. Route through a virtual cable ([VB-Cable](https://vb-audio.com/Cable/)):

1. Set Windows default playback to `CABLE Input`
2. `python live.py -i "CABLE Output" -o "Speakers"`

Original audio goes to the cable, you only hear the separated stem.

### File conversion

Handled entirely by the upstream CLI, same models and cache:

```sh
audio-separator song.mp3 --model_filename vocals_mel_band_roformer.ckpt
audio-separator --list_models --list_filter=vocals
```

## Install

1. Install [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) ([all platforms](https://ffmpeg.org/download.html)) and add to PATH
2. Install [VB-Audio Cable](https://vb-audio.com/Cable/) (optional, recommended for routing system audio, see [Streaming system audio](#streaming-system-audio))
3. Install Python dependencies:
```sh
pip install -r requirements.txt
```
4. For GPU acceleration (recommended), install the CUDA torch build:
```sh
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
RTX 50xx (Blackwell) requires cu128+. CPU-only works but ~10x slower, check the startup benchmark warning.

## Tuning

Constants at the top of `live.py`:

| Parameter           | Default | Description                                                     |
| ------------------- | ------- | --------------------------------------------------------------- |
| `blocksize`         | 4410    | Stream frame size, 0.1s at 44.1kHz                              |
| `window_size`       | 16      | Frames per inference window, quality degrades under ~1.5s       |
| `overlap_size`      | 1       | Frames trimmed from window edges to hide seam artifacts         |
| `initial_wait_size` | 12      | Frames buffered before playback starts, raise if audio stutters |

## How it works

Incoming frames accumulate in a pending buffer. Every `window_size - 2 * overlap_size` frames, a window is assembled from the previous window's tail (overlap) plus fresh frames and sent to the model on a background thread. Only the middle of each separated window is queued for playback, the overlapping edges are discarded to mask block-boundary artifacts. Frames consumed and produced per cycle are equal, so the pipeline stays balanced.

Roformer models default to ~8s inference chunks and reject shorter input, so `live.py` overrides the model segment size to fit the streaming window.

## Benchmark

RTX 5080, 1.6s window:

| Model                                      | Inference per window | Headroom                 |
| ------------------------------------------ | -------------------- | ------------------------ |
| `vocals_mel_band_roformer.ckpt` (SDR 12.6) | 0.10s                | 16x faster than realtime |
| `UVR-MDX-NET-Inst_HQ_3.onnx`               | 0.18s                | 9x faster than realtime  |

End-to-end latency ~1.7s: window fill 1.4s + inference + one frame.

## Credits

- [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator): model loading and inference
- [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser): streaming buffer pattern
- [Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui): model ecosystem

## License

[MIT](LICENSE)
