import av
import numpy as np

def read_video_pyav(video_path, frame=8):
    '''
    Decode the video with PyAV decoder.

    Args:
        video_path:
        frame:

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''

    container = av.open(video_path)

    # sample uniformly frames from the video 
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / frame).astype(int)

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])