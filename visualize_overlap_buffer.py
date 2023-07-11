
from collections import deque
import numpy as np
import time 

'''
Simple test to visualize the overlapping buffer
'''

def process():
    print(f'processing_buffer: {processing_buffer}')
    out = [x for x in processing_buffer]
    available_chunks = len(out)
    [processed_buffer.append(x[0]) for x in np.array_split(out, available_chunks)[overlap_size:-overlap_size]]
    overlap_buffer.append(processing_buffer.pop())
    overlap_buffer.append(processing_buffer.pop())

window_size = 8
overlap_size = 1

pending_buffer = deque(maxlen=window_size)
processing_buffer = deque(maxlen=window_size + (overlap_size*2))
processed_buffer = deque(maxlen=window_size)
overlap_buffer = deque(maxlen=overlap_size*2)
counter = 0

while True:
    pending_buffer.append(counter)
    counter += 1
    
    if len(pending_buffer) >= window_size-2:
        processing_buffer.clear()

        while len(overlap_buffer) > 0:
            processing_buffer.append(overlap_buffer.pop())

        for i in range(window_size-2):
            processing_buffer.append(pending_buffer.popleft())

        process()
    
    if len(processed_buffer) > 0:
        print(f'processed_buffer: {processed_buffer}')
        processed_buffer.clear()

    time.sleep(0.2)
