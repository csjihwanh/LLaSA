import torch
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from model.llasa_arch import LLaSA
from model.llasa_processor import LlavaNextVideoProcessor

def build_llasa():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = LlavaNextVideoProcessor.from_pretrained("/workspace/LLaSA/checkpoints/LLaVA-NeXT-Video-7B-hf", cache_dir='/workspace/LLaSA/checkpoints')
    model = LLaSA.from_pretrained(
        "/workspace/LLaSA/checkpoints/LLaVA-NeXT-Video-7B-hf",
        quantization_config=quantization_config,
        device_map='auto',
        cache_dir = '/workspace/LLaSA/checkpoints'
    )

    return model, processor

