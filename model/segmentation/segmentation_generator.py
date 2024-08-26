import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from .sam2.build_sam import build_sam2_video_predictor

from config.config import load_configs

class SegmentationGenerator(nn.Module):
    def __init__(self):
        
        config = load_configs()

        sam2_checkpoint = config.segmentation.checkpoint
        model_cfg = config.segmentation.model_cfg
        
        self.device = config.device

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)

        self.debug_dir = config.debug_dir

    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _save_annotated_frame(self, video_dir, frame_name, frame_idx):

        debug_dir = ''
            
        # Create a new figure for the frame
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {frame_idx}")
        
        # Load and display the frame
        frame_path = os.path.join(debug_dir, 'debug_img')
        plt.imshow(Image.open(os.path.join(video_dir, frame_name)))
        
        # Overlay segmentation masks on the frame
       # for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #    self._show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        
        # Save the figure to the debug directory
        #save_path = os.path.join(self.debug_dir, f"annotated_frame_{out_frame_idx}.png")
        #plt.savefig(save_path)
        #plt.close()  # Close the plot to free up memory

    # Hook function to capture intermediate outputs
    def hook_fn(self, module, input, output):
        self.outputs['sam_mask_decoder_transformer'] = output

    # Method to register the hook conditionally
    def register_hook(self):
        # Register the hook on the TwoWayTransformer module within sam_mask_decoder
        self.hook_handle = self.predictor.sam_mask_decoder.transformer.register_forward_hook(self.hook_fn)

    # Method to remove the hook if needed
    def remove_hook(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()

    def inference_video(self, video_path, interact_frame, bbox, debug=True, intermediate_result=False):
        # video path must be a directory that contains all jpeg files of the video 
        # mask format: np.float32((x_min, y_min, x_max, y_max))
        
        if intermediate_result:
            self.register_hook()
        
        inference_state = self.predictor.init_state(video_path = video_path)
        self.predictor.reset_state(inference_state)

        ann_obj_id = 1

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=interact_frame,
            obj_id=ann_obj_id,
            box=bbox,
        )

        # result starts from interact_frame 
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results(out_obj_id, out_mask)
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }


        if intermediate_result:
            # Remove the hook if it's no longer needed
            self.remove_hook()
            # Return the captured intermediate activation
            return video_segments, self.outputs['sam_mask_decoder_transformer']
        
        return video_segments

