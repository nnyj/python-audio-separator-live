import sounddevice as sd
import numpy as np
from audio_separator import Separator
from collections import deque
import logging
import threading
# Extra buffer for CPU mode

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

def callback(indata, outdata, frames, time, status):
    global initial_wait
    if status:
        print(status)
    pending_buffer.append(indata.copy())

    if len(pending_buffer) >= window_size:
        for i in range(window_size):
            processing_buffer.append(pending_buffer.popleft())

        separate_thread = threading.Thread(target=separate_audio)
        separate_thread.start()

    print(len(processed_buffer))
    if len(processed_buffer) == 0: initial_wait = True
    if len(processed_buffer) > initial_delay: initial_wait = False
    if len(processed_buffer) > 0 and not initial_wait:
        outdata[:] = processed_buffer.popleft()

def main():
    global streamer
    streamer = Separator(None, model_name='UVR-MDX-NET-Inst_Main', model_file_dir="/tmp/audio-separator-models/", use_cuda=False,
    log_level=logging.DEBUG)

    try:
        with sd.Stream(device=("CABLE-A Output (VB-Audio Cable , MME", "SteelSeries Sonar - Gaming (Ste, MME"), samplerate=samplerate, channels=2, callback=callback, blocksize=blocksize):
            print(f'Latency: {(blocksize*window_size/samplerate):.2f}s, Frame: {(blocksize/samplerate):.2f}s')
            print('press Return to quit')
            input()
    except KeyboardInterrupt:
        print("Stopping")

if __name__ == "__main__":
    main()