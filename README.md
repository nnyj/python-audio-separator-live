# Audio Separator in Real Time

This was quickly pieced together for fun. The audio separation quality is not the best, but it might work for your use-case.

Any feedback or pull requests are appreciated.

# Installation

Clone this repo.

# Usage

1. cd to the project root directory and run `python live.py`

Command-line arguments:
```
-i or --in: Input device
-o or --out: Output device
log_level: (Optional) Logging level, e.g. info, debug, warning. Default: INFO
model_name: (Optional) The name of the model to use for separation. Default: UVR_MDXNET_KARA_2
model_file_dir: (Optional) Directory to cache model files in. Default: /tmp/audio-separator-models/
use_cuda: (Optional) Flag to use Nvidia GPU via CUDA for separation if available. Default: False
```

# Hardware requirements

GPU mode is necessary to perform the inference in near realtime.

Make sure to install the onnxruntime-gpu:
`pip install onnxruntime-gpu`

## Benchmark results:

| CPU / GPU            | model_run() speed | window_size | overlap_size | initial_wait_size | block_size | sample_rate | Theoretical latency |
| -------------------- | ----------------- | ----------- | ------------ | ----------------- | ---------- | ----------- | ------------------- |
| i7-12700K & RTX 3090 | 0.04s             | 20          | 1            | 0                 | 4000       | 48000       | 1.75s               |
| i7-12700K            | 0.82s             | 20          | 1            | 16                | 4000       | 48000       | 3.08s               |

# Configurable parameters

| Parameters        | Suggested values | Description                                                                                                 |
| ----------------- | ---------------- | ----------------------------------------------------------------------------------------------------------- |
| window_size       | 16-32            | Processing window for inference, recommended at least 1.5 seconds                                           |
| overlap_size      | 1-4              | How many frames to keep before and after the processing window to reduce artifacts                          |
| initial_wait      | True/False       | Use True for CPU, False for GPU                                                                             |
| initial_wait_size | 16               | Initial frames to buffer for slower CPUs, duration should be longer than time needed to execute model_run() |
| blocksize         | 4000             | The rate to call the callback function of sounddevice                                                       |
| use_threading     | True/False       | Use True for CPU, False for GPU (Introduces additional ~3s delay)                                           |

# Features

- Real-time audio separation using any of the MDX-NET single model.
- Approximately 1-5 seconds latency depending on hardware.
- No ensemble support yet.

# Credits

- [karaokenerds](https://github.com/karaokenerds) - Author of [python-audio-separator](https://github.com/karaokenerds/python-audio-separator), a python package based on [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) by [Anjok07](https://github.com/Anjok07).
- [facebookresearch](https://github.com/facebookresearch) - Author of [denoiser](https://github.com/facebookresearch/denoiser). Copied code for implementing real-time streaming via sounddevice.