import numpy as np


def average_sample(frames_len, num_frames):
    sample_idx = []
    for length in range(frames_len):
        temp_idx = []
        if length == 0:
            temp_idx = []
        elif length <= num_frames:
            extend_len = num_frames - length
            for i in range(extend_len//2):
                temp_idx.append(0)
            temp_idx.extend(range(length))
            for i in range(extend_len-extend_len//2):
                temp_idx.append(length-1)
        else:
            temp_idx = np.linspace(0, length-1, num_frames, dtype=np.int32).tolist()
        sample_idx.append(temp_idx)
    return sample_idx