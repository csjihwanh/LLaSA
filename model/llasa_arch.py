"""
This code referenced
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L386
"""

import torch
from torch import nn
from typing import List, Optional, Tuple, Union

from transformers import (
    BitsAndBytesConfig, 
    LlavaNextVideoForConditionalGeneration, 
    LlavaNextVideoProcessor, 
    LlavaNextVideoConfig, 
    
)
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoCausalLMOutputWithPast,
    LLAVA_NEXT_VIDEO_INPUTS_DOCSTRING
)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from transformers.cache_utils import Cache

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaNextVideoConfig"

class LLaSA(LlavaNextVideoForConditionalGeneration):
    def __init__(
            self,
            config: LlavaNextVideoConfig
        ):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(LLAVA_NEXT_VIDEO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaNextVideoCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        seg_tokens = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaNextVideoCausalLMOutputWithPast]:
        r"""
        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, image_size, image_size)):
                The tensors corresponding to the input videos. Pixel values can be obtained using
                [`AutoImageProcessor`]. See [`LlavaNextVideoVideoProcessor.__call__`] for details. [`LlavaProcessor`] uses
                [`LlavaNextVideoVideoProcessor`] for processing videos.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> import av
        >>> from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

        >>> model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", device_map="auto)
        >>> processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

        >>> prompt = "USER: <video>\nWhy is this video funny? ASSISTANT:"
        >>> video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
        >>> container = av.open(video_path)

        >>> # sample uniformly 8 frames from the video (model was trained with 32 frames per video, but this video is short)
        >>> total_frames = container.streams.video[0].frames
        >>> indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        >>> clip = read_video_pyav(container, indices)
        >>> inputs_video = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)

        >>> # load an image to generate from an image
        >>> prompt = "USER:<image>\nWhat is shown in this image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs_image = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        >>> # Generate from video
        >>> generate_ids = model.generate(**inputs_video, max_length=50)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:\nWhy is this video funny? ASSISTANT: The humor in this video comes from the unexpected and endearing sight of a baby wearing glasses and (...)"

        >>> # Generate from image
        >>> generate_ids = model.generate(**inputs_image, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER: \nWhat's the content of the image? ASSISTANT: The image shows a red stop sign on a pole, with a traditional Chinese archway (...)"
        ```"""
        #import pdb; pdb.set_trace()
        if seg_tokens is not None:
            print('got seg!!:', seg_tokens.shape, position_ids)
        if pixel_values_videos is not None:
            print('got vid!!:', pixel_values_videos.shape, position_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        self.vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if (pixel_values is not None or pixel_values_videos is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image/video tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            img_token_count = (input_ids == self.config.image_token_index).sum(1).max()
            video_token_count = (input_ids == self.config.video_token_index).sum(1).max()
            inputs_expanded = (
                img_token_count < self.config.image_seq_length and video_token_count < self.config.video_seq_length
            )
            pixels_present = input_ids.shape[-1] == 1 and pixel_values is not None and pixel_values_videos is not None
            legacy_processing = inputs_expanded or pixels_present

        image_features = feature_lens = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self._get_image_features(pixel_values, image_sizes)
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline,
            )

        video_features = video_feature_lens = None
        if pixel_values_videos is not None and pixel_values_videos.size(0) > 0:
            video_features = self._get_video_features(pixel_values_videos)
            video_features = [feature.flatten(0, 1) for feature in video_features]
            video_feature_lens = [feature.size(0) for feature in video_features]
            video_features = torch.cat(video_features, dim=0)
            video_feature_lens = torch.tensor(video_feature_lens, dtype=torch.long, device=video_features.device)

            if legacy_processing:
                logger.warning_once(
                    "Expanding inputs for image.video tokens in LLaVa-NeXT-Video should be done in processing. "
                    "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                    "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                    "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                )
                if input_ids.shape[1] != 1:
                    iterator = (
                        (image_features, feature_lens, self.config.image_token_index),
                        (video_features, video_feature_lens, self.config.video_token_index),
                    )
                    for features, lens, special_token in iterator:
                        if features is not None:
                            (
                                inputs_embeds,
                                attention_mask,
                                position_ids,
                                labels,
                                input_ids,
                            ) = self._merge_input_ids_with_image_features(
                                features,
                                lens,
                                inputs_embeds,
                                input_ids,
                                attention_mask,
                                position_ids,
                                labels=labels,
                                image_token_index=special_token,
                            )
                else:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)
                    # Get the target length
                    target_length = input_ids.shape[1]
                    past_length = first_layer_past_key_value.shape[-1]
                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], past_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]
                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                    attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

            # TODO: @raushan retain only the new behavior after v4.47
            else:
                if image_features is not None:
                    special_image_mask = (
                        (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    )
                    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

                if video_features is not None:
                    special_image_mask = (
                        (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    )
                    video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, video_features)
        
        ### concatenate segmentation token

        if seg_tokens is not None:

            batch_size, seg_token_length, embedding_dim = seg_tokens.shape
            assert embedding_dim == inputs_embeds.shape[-1], "Embedding dimensions must match"

            inputs_embeds = torch.cat((inputs_embeds, seg_tokens), dim=1)

            # Update attention mask to account for the new segmentation tokens
            seg_mask = torch.ones(batch_size, seg_token_length, device=inputs_embeds.device)
            attention_mask = torch.cat((attention_mask, seg_mask), dim=1)

            # Update position_ids accordingly
            # If the segmentation tokens are added at the end, extend position_ids
            if position_ids is not None:
                seg_position_ids = (
                    position_ids[:, -1:] + torch.arange(1, seg_token_length + 1, device=position_ids.device).unsqueeze(0)
                )
                position_ids = torch.cat((position_ids, seg_position_ids), dim=1)
            else:
                position_ids = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0).repeat(batch_size, 1)

        ### segmentation addition part done 

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaNextVideoCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        seg_tokens=None,
        image_sizes=None,
        attention_mask=None,
        **kwargs,
    ):
        #import pdb; pdb.set_trace()
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids or self.config.video_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]

            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "seg_tokens":seg_tokens,
                "image_sizes": image_sizes,
            }
        )
        return model_inputs


