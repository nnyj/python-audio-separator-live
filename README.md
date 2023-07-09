# Audio Separator in Real Time

This was quickly pieced together for fun. The audio separation quality is not the best, but it might work for your use-case.

Any feedback or pull requests are appreciated.

# Installation

Clone this repo.

# Usage

1. cd to the project root directory and run `python live.py`

Command-line arguments:
- `-i` or `--in`: Input device
- `-o` or `--out`: Output device

# Hardware requirements

GPU mode is necessary to perform the inferenece in near realtime.

Make sure to install the onnxruntime-gpu:
`pip install onnxruntime-gpu`

## Benchmark results:

| model_run() | window_size | frame_length | sample_rate | CPU / GPU            | Remarks           |
| ----------- | ----------- | ------------ | ----------- | -------------------- | ----------------- |
| 0.05s       | 8           | 5000         | 48000       | i7-12700K & RTX 3090 |                   |
| 0.82s       | 8           | 5000         | 48000       | i7-12700K            | Basically useless |

# Configurable parameters

- window_size: Play around with values between 8 to 16. The total latency would be multiplicative of the frame_length.
- frame_length: Play around with values between 5000 to 10000.

The middle half of the predicted input buffer was taken as the output to minimize most audible artifacts. Hence, it may not be the most efficient use of resource.

# Features

- Real-time audio separation using any of the MDX-NET single model.
- Approximately 1-2 seconds latency depending on hardware.
- No ensemble support yet.

# Credits

- [karaokenerds](https://github.com/karaokenerds) - Author of [python-audio-separator](https://github.com/karaokenerds/python-audio-separator), a python package based on [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) by [Anjok07](https://github.com/Anjok07).
- [facebookresearch](https://github.com/facebookresearch) - Author of [denoiser](https://github.com/facebookresearch/denoiser). Copied code for implementing real-time streaming via sounddevice.