import sounddevice as sd
import numpy as np
from collections import deque
from audio_separator import Separator
import logging
import threading
  
samplerate=48000
blocksize = 4000
window_size = 32
pending_buffer = deque(maxlen=window_size*2)
processed_buffer = deque(maxlen=window_size*3)
processing_buffer = deque(maxlen=window_size)
initial_delay = window_size//2 # Adjust this for slower CPUs
initial_wait = True

def separate_audio():
  _, _, out = streamer.separate(np.concatenate(processing_buffer, axis=0))
  [processed_buffer.append(x) for x in np.array_split(out, window_size)]

def main():
  global streamer
  streamer = Separator(None,
    model_name='UVR-MDX-NET-Inst_Main',
    model_file_dir="/tmp/audio-separator-models/",
    use_cuda=False,
    log_level=logging.DEBUG)

  stream_in = sd.InputStream( device="CABLE-A Output (VB-Audio Cable , MME", samplerate=samplerate, channels=2)
  stream_out = sd.OutputStream( device="SteelSeries Sonar - Gaming (Ste, MME", samplerate=samplerate, channels=2)

  stream_in.start()
  stream_out.start()

  while True:
    try:
      frame, overflow = stream_in.read(blocksize)
      pending_buffer.append(frame)

      if len(pending_buffer) >= window_size:
        for i in range(window_size):
          processing_buffer.append(pending_buffer.popleft())

        separate_thread = threading.Thread(target=separate_audio)
        separate_thread.start()

      print(len(processed_buffer))
      if len(processed_buffer) == 0: initial_wait = True
      if len(processed_buffer) > initial_delay: initial_wait = False
      if len(processed_buffer) > 0 and not initial_wait:
        underflow = stream_out.write(processed_buffer.popleft())
    except KeyboardInterrupt:
      print("Stopping")
      break
  stream_out.stop()
  stream_in.stop()

if __name__ == "__main__":
  main()