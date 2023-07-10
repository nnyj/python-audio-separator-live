import sounddevice as sd
import numpy as np
from audio_separator import Separator
from collections import deque
import logging

samplerate=48000
blocksize = 4000
window_size = 16
pending_buffer = deque(maxlen=window_size)
processed_buffer = deque(maxlen=window_size)

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    pending_buffer.append(indata.copy()) # The copy() is necessary

    if len(pending_buffer) >= window_size:
        _, _, out = streamer.separate(np.concatenate(pending_buffer, axis=0))
        [processed_buffer.append(x) for x in np.array_split(out, window_size)]
        pending_buffer.clear()

    if len(processed_buffer) > 0:
        outdata[:] = processed_buffer.popleft()

def main():
    global streamer
    streamer = Separator(None, model_name='UVR-MDX-NET-Inst_Main', model_file_dir="/tmp/audio-separator-models/", use_cuda=True,
    log_level=logging.DEBUG)

    try:
        with sd.Stream(device=("CABLE-A Output (VB-Audio Cable , MME", "SteelSeries Sonar - Gaming (Ste, MME"), samplerate=samplerate, channels=2, callback=callback, blocksize=blocksize):
            print(f'Latency: {(blocksize*window_size/samplerate):.2f}s')
            print('press Return to quit')
            input()
    except KeyboardInterrupt:
        print("Stopping")

if __name__ == "__main__":
    main()