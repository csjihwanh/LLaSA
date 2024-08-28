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

from model.projector.seg_projector import seg_projector_builder

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaNextVideoConfig"

class LLaSA(LlavaNextVideoForConditionalGeneration):
    def __init__(
            self,
            config: LlavaNextVideoConfig
        ):
        super().__init__(config)

        self.seg_projector = seg_projector_builder()

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

        SEG_TOKEN_INDEX = 32002

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
            seg_features = self.seg_projector(seg_tokens)
            if torch.any(input_ids == SEG_TOKEN_INDEX):
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    labels,
                    input_ids,
                ) = self._merge_input_ids_with_seg_features(
                    seg_features,
                    seg_features.shape[1],
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels
                )

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

    def _merge_input_ids_with_seg_features(
        self,
        seg_features,
        feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids,
        labels,
        seg_token_index=32002,
        ignore_index=-100,
    ):
        """
        Merge input_ids with seg features into final embeddings by replacing the seg token with corresponding seg_features.

        Args:
            seg_features (`torch.Tensor` of shape `(1, 256, 4096)`):
                Pre-processed segmentation features to merge into the sequence.
            feature_lens (`torch.LongTensor`):
                Not needed here since there's only one seg token.
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Embeddings for the input tokens.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Token IDs of the inputs, containing one special seg token to be replaced.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Attention mask for the input tokens.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Position IDs for the input sequence tokens.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for training purposes, if applicable.
            seg_token_index (`int`):
                Token index that represents the special segmentation token.
            ignore_index (`int`):
                Index to ignore in loss calculation (typically for padding).

        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels
        """
        PAD_INDEX = -1 

        # Determine sizes
        batch_size, seq_length, embed_dim = inputs_embeds.shape
        seg_length = seg_features.size(1)

        # Prepare new tensor sizes
        new_seq_length = seq_length + seg_length - 1  # Since we replace one token with `seg_length` embeddings

        # Create final tensors with updated sequence length
        final_embedding = torch.zeros((batch_size, new_seq_length, embed_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        final_attention_mask = torch.zeros((batch_size, new_seq_length), dtype=attention_mask.dtype, device=attention_mask.device)
        final_input_ids = torch.full((batch_size, new_seq_length), fill_value=PAD_INDEX, dtype=input_ids.dtype, device=input_ids.device)

        # Labels initialization if they exist
        final_labels = torch.full((batch_size, new_seq_length), fill_value=ignore_index, dtype=labels.dtype, device=labels.device) if labels is not None else None

        for batch_idx in range(batch_size):
            # Locate the seg token in the sequence
            seg_token_pos = (input_ids[batch_idx] == seg_token_index).nonzero(as_tuple=True)[0].item()

            # Split the embeddings into parts before and after the seg token
            before_seg = inputs_embeds[batch_idx, :seg_token_pos, :]
            after_seg = inputs_embeds[batch_idx, seg_token_pos + 1:, :]

            # Construct the final embeddings
            final_embedding[batch_idx] = torch.cat((before_seg, seg_features[0], after_seg), dim=0)

            # Update attention mask (set to 1 for all valid tokens)
            final_attention_mask[batch_idx, :before_seg.size(0)] = 1
            final_attention_mask[batch_idx, before_seg.size(0):before_seg.size(0) + seg_length] = 1
            final_attention_mask[batch_idx, before_seg.size(0) + seg_length:before_seg.size(0) + seg_length + after_seg.size(0)] = 1

            # Update input_ids with original ids (if needed)
            final_input_ids[batch_idx, :before_seg.size(0)] = input_ids[batch_idx, :seg_token_pos]
            final_input_ids[batch_idx, before_seg.size(0) + seg_length:before_seg.size(0) + seg_length + after_seg.size(0)] = input_ids[batch_idx, seg_token_pos + 1:]

            # Update labels if they are provided
            if final_labels is not None:
                final_labels[batch_idx, :before_seg.size(0)] = labels[batch_idx, :seg_token_pos]
                final_labels[batch_idx, before_seg.size(0) + seg_length:before_seg.size(0) + seg_length + after_seg.size(0)] = labels[batch_idx, seg_token_pos + 1:]

        # Recalculate position_ids if necessary (simple consecutive order)
        position_ids = torch.arange(new_seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, new_seq_length)

        return final_embedding, final_attention_mask, position_ids, final_labels, final_input_ids


    def _merge_input_ids_with_image_features(
        self,
        image_features,
        feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
        image_token_index=None,
        ignore_index=-100,
    ):
        """
        Merge input_ids with with image features into final embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`):
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                :abels need to be recalculated to support training (if provided)
            image_token_index (`int`, *optional*)
                Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
            ignore_index (`int`, *optional*)
                Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (8)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but image token sizes are different, then cannot infer left or right padding
                ```python
                cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
                chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)
                prompts = [
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                ]
                inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                    chart_img has 2634 tokens, while cat_img has 2340 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
        ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

        with torch.no_grad():
            # ! in llava 1.6, number of patches is variable
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = True if not self.training else False
            if batch_size > 1 and not self.training:
                if _left_padding and not _right_padding:
                    left_padding = True
                elif not _left_padding and _right_padding:
                    left_padding = False
                elif not _left_padding and not _right_padding:
                    # both side is 1, so cannot tell
                    left_padding = self.padding_side == "left"
                else:
                    # invalid attention_mask
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens = feature_lens.to(input_ids.device)
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_embed_dim), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                max_embed_dim = max_embed_dim.to(target_device)
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(torch.float16).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids, final_labels, final_input_ids
