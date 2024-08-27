import json
import os
import numpy as np
from tqdm import tqdm
import torch

from config.config import load_configs
from segmentation_generator import SegmentationGenerator


file_path = 'dataset/A2D/output_video_info.json'
video_path_base = 'dataset/A2D/clips320jpeg'
store_path = 'dataset/A2D/a2d_sam_tensor'

config = load_configs()

seg_generator = SegmentationGenerator(config)

with open(file_path, 'r') as file:
    data = json.load(file)

for idx, (key, value) in enumerate(tqdm(data.items(), total=len(data), desc="Processing")):
    video = value['video']
    bbox_frame = value['bbox_frame']
    bbox = np.float32(value['bbox'])
    caption = value['caption']

    store_file_path = os.path.join(store_path, f'{key}.pt')
    
    # Check if the tensor file already exists
    if os.path.exists(store_file_path):
        print(f"Tensor for {key} already exists. Skipping processing.")
        continue  # Skip to the next item
    
    video_path = os.path.join(video_path_base, video)

    _, result = seg_generator(video_path, bbox_frame, bbox, intermediate_result=True)

    torch.save(result, store_file_path)



