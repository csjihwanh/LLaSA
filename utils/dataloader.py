import json
import torch
import random
from torch.utils.data import Dataset

from utils.utils import read_video_pyav
import datasets

# Define the custom Dataset class
class Dataset_A2D(Dataset):
    def __init__(self, start_idx, end_idx, processor):
        
        self.sam_tensor_path_base='/workspace/LLaSA/dataset/A2D/a2d_sam_tensor'
        self.video_path_base ='/workspace/LLaSA/dataset/A2D/clips320H'
        
        json_file_path='/workspace/LLaSA/dataset/A2D/output_video_info.json'

        self.processor = processor

        # Load data from JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        self.keys = list(data.keys())[start_idx:end_idx] 
        self.data = {key: data[key] for key in self.keys} 

    def _instruction_generator(self):
        instructions = [
            "Describe the main action being performed by the actor in the video. Use the segmentation information to provide more details about the actor and their movements.",
            "Describe how the actor interacts with other objects in the video. Use the segmentation data to identify the key objects involved and detail their interaction.",
            "Describe the scene and the behavior of the actor within it. Use segmentation to emphasize the actor and provide details about their actions.",
            "Describe the interaction between multiple actors in the video. Use the segmentation information to highlight these interactions and describe their actions.",
            "Describe the movements and changes in the scene involving the actor. Use segmentation to focus on the actor and describe the specific changes observed."
        ]
        return random.choice(instructions)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        
        # Extract data
        caption = item['caption']
        video_id = item['video']

        video_path = f"{self.video_path_base}/{video_id}.mp4"
        sam_tensor_path = f"{self.sam_tensor_path_base}/{key}.pt" 

        video = read_video_pyav(video_path)
        seg = torch.load(sam_tensor_path)

        instruction = self._instruction_generator()
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "video"},
                    {"type": "seg"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor([prompt], videos=[video], seg=seg, labels=caption, padding=True, return_tensors="pt")
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.float16)
        inputs['seg_tokens'] = inputs['seg_tokens'].to(torch.float16)
        #inputs = {key: value.squeeze(0) for key, value in inputs.items()}

        return inputs

def get_a2d_hf_dataset(processor, start_idx, end_idx):
    a2d_torch = Dataset_A2D(
        processor=processor,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    return datasets.Dataset.from_list(a2d_torch)