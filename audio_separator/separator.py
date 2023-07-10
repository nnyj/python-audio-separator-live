import os
import warnings
import hashlib
import json
import logging
import warnings
import wget
import torch
import numpy as np
import onnxruntime as ort
from audio_separator.utils import spec_utils
import time

class Separator:
  def __init__(
    self,
    audio_file_path,
    model_name="UVR_MDXNET_KARA_2",
    model_file_dir="/tmp/audio-separator-models/",
    output_dir=None,
    use_cuda=False,
    log_level=logging.DEBUG,
    log_formatter=None,
  ):
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(log_level)
    self.log_level = log_level
    self.log_formatter = log_formatter

    self.log_handler = logging.StreamHandler()

    if self.log_formatter is None:
      self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    self.log_handler.setFormatter(self.log_formatter)
    self.logger.addHandler(self.log_handler)

    self.logger.debug(
      f"Separator instantiating with input file: {audio_file_path}, model_name: {model_name}, output_dir: {output_dir}, use_cuda: {use_cuda}"
    )

    self.model_name = model_name
    self.model_file_dir = model_file_dir
    self.output_dir = output_dir
    self.use_cuda = use_cuda
    self.audio_file_path = audio_file_path
    # self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

    self.model_name = model_name
    self.model_url = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{self.model_name}.onnx"
    self.model_data_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json"

    self.wav_type_set = "PCM_16"
    self.is_normalization = False
    self.is_denoise = False

    self.chunks = 0
    self.margin = 48000
    self.adjust = 1
    self.dim_c = 4
    self.hop = 1024

    self.primary_source = None
    self.secondary_source = None

    warnings.filterwarnings("ignore")
    self.cpu = torch.device("cpu")
    if self.use_cuda:
      self.logger.debug("Running in GPU mode")
      self.device = torch.device("cuda")
      self.run_type = ["CUDAExecutionProvider"]
    else:
      self.logger.debug("Running in CPU mode")
      self.device = torch.device("cpu")
      self.run_type = ["CPUExecutionProvider"]
    self.logger.info(f'ort.get_device: {ort.get_device()}')

    # Custom
    # frame_length originally 256, set between 5000-10000 to prevent audio skipping
    self.frame_length = 5000
    self.frames = 0
    self.total_length = self.frame_length
    self.total_time = 0
    self.stride = self.frame_length

    # Initialize the model
    model_path = os.path.join(self.model_file_dir, f"{self.model_name}.onnx")
    if not os.path.isfile(model_path):
      self.logger.debug(f"Model not found at path {model_path}, downloading...")
      wget.download(self.model_url, model_path)

    self.logger.debug("Reading model settings...")

    model_hash = self.get_model_hash(model_path)
    self.logger.debug(f"Model {model_path} has hash {model_hash} ...")

    model_data_path = os.path.join(self.model_file_dir, "model_data.json")
    if not os.path.isfile(model_data_path):
      self.logger.debug(f"Model data not found at path {model_data_path}, downloading...")
      wget.download(self.model_data_url, model_data_path)

    model_data_object = json.load(open(model_data_path))
    model_data = model_data_object[model_hash]

    self.compensate = model_data["compensate"]
    self.dim_f = model_data["mdx_dim_f_set"]
    self.dim_t = 2 ** model_data["mdx_dim_t_set"]
    self.n_fft = model_data["mdx_n_fft_scale_set"]
    self.primary_stem = model_data["primary_stem"]
    self.secondary_stem = "Vocals" if self.primary_stem == "Instrumental" else "Instrumental"

    self.logger.debug(
      f"Set model data values: compensate = {self.compensate} primary_stem = {self.primary_stem} dim_f = {self.dim_f} dim_t = {self.dim_t} n_fft = {self.n_fft}"
    )
  
    self.logger.debug("Loading model...")
    self.ort_ = ort.InferenceSession(model_path, providers=self.run_type)
    self.model_run = lambda spek: self.ort_.run(None, {"input": spek.cpu().numpy()})[0]
    self.initialize_model_settings()
  
  def reset_time_per_frame(self):
    # Copied from demucs.py
    self.total_time = 0
    self.frames = 0

  @property
  def time_per_frame(self):
    # Copied from demucs.py
    return self.total_time / self.frames

  def get_model_hash(self, model_path):
    try:
      with open(model_path, "rb") as f:
        f.seek(-10000 * 1024, 2)
        return hashlib.md5(f.read()).hexdigest()
    except:
      return hashlib.md5(open(model_path, "rb").read()).hexdigest()

  def separate(self, input_wav=None):
    begin = time.time()
    # self.logger.debug(f'separate(): {begin}, frames: {self.frames}')
    self.frames += 1

    # self.logger.info("Running inference...")
    mix, raw_mix, samplerate = prepare_mix(input_wav, self.chunks, self.margin)
    # self.logger.info("Demixing...")
    source = self.demix_base(mix)
    source = source[0]

    self.primary_source = spec_utils.normalize(self.logger, source, self.is_normalization).T
    self.primary_source = np.asarray(self.primary_source, order='C')

    self.total_time += time.time() - begin
    # self.logger.debug(f'self.total_time: {self.total_time}')

    # return None, None, input_wav
    return None, None, self.primary_source

  def initialize_model_settings(self):
    self.n_bins = self.n_fft // 2 + 1
    self.trim = self.n_fft // 2
    self.chunk_size = self.hop * (self.dim_t - 1)
    self.window = torch.hann_window(window_length=self.n_fft, periodic=False).to(self.device)
    self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins - self.dim_f, self.dim_t]).to(self.device)
    self.gen_size = self.chunk_size - 2 * self.trim

  def initialize_mix(self, mix):
    mix_waves = []
    n_sample = mix.shape[1]
    pad = self.gen_size - n_sample % self.gen_size
    mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)
    i = 0
    while i < n_sample + pad: # This loop only runs once if gen_size (256000) bigger than mix.shape
      waves = np.array(mix_p[:, i : i + self.chunk_size])
      mix_waves.append(waves)
      i += self.gen_size

    mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

    return mix_waves, pad

  def demix_base(self, mix):
    chunked_sources = []
    for slice in mix:
      sources = []
      tar_waves_ = []
      mix_p = mix[slice]
      mix_waves, pad = self.initialize_mix(mix_p)
      mix_waves = mix_waves.split(1)
      pad = -pad
      with torch.no_grad():
        for mix_wave in mix_waves:
          tar_waves = self.run_model(mix_wave)
          tar_waves_.append(tar_waves)
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :pad]
        start = 0 if slice == 0 else self.margin
        end = None if slice == list(mix.keys())[::-1][0] or self.margin == 0 else -self.margin
        sources.append(tar_waves[:, start:end] * (1 / self.adjust))
      chunked_sources.append(sources)
    sources = np.concatenate(chunked_sources, axis=-1)

    return sources
  
  def run_model(self, mix):
    spek = self.stft(mix.to(self.device)) * self.adjust
    spek[:, :, :3, :] *= 0
    begin = time.time()
    spec_pred = self.model_run(spek)
    end = time.time() - begin
    self.logger.debug(f'model_run(): {end:.3f}s')
    output = (
      self.istft(torch.tensor(spec_pred).to(self.device))
      .to(self.cpu)[:, :, self.trim : -self.trim]
      .transpose(0, 1)
      .reshape(2, -1)
      .numpy()
    )
    return output

  def stft(self, x):
    x = x.reshape([-1, self.chunk_size])
    x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
    x = torch.view_as_real(x)
    x = x.permute([0, 3, 1, 2])
    x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
    return x[:, :, : self.dim_f]

  def istft(self, x, freq_pad=None):
    freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
    x = torch.cat([x, freq_pad], -2)
    x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
    x = x.permute([0, 2, 3, 1])
    x = x.contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
    return x.reshape([-1, 2, self.chunk_size])

def prepare_mix(mix, chunk_set, margin_set):
  samplerate = 48000
  mix = mix.T
  samples = mix.shape[-1]
  margin = margin_set
  chunk_size = chunk_set * 48000

  assert not margin == 0, "margin cannot be zero!"

  if margin > chunk_size:
    margin = chunk_size
  if chunk_set == 0 or samples < chunk_size:
    chunk_size = samples

  segmented_mix = {}
  counter = -1
  for skip in range(0, samples, chunk_size):
    counter += 1
    s_margin = 0 if counter == 0 else margin
    end = min(skip + chunk_size + margin, samples)
    start = skip - s_margin
    segmented_mix[skip] = mix[:, start:end]  # Avoid unnecessary copy

    if end == samples:
      break

  raw_mix = segmented_mix.copy()  # Create a shallow copy instead of calling the function again

  return segmented_mix, raw_mix, samplerate