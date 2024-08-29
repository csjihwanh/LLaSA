import torch
from transformers import BitsAndBytesConfig
from model.llasa_arch import LLaSA
from model.llasa_processor import LlavaNextVideoProcessor, LlasaDataCollatorWithPadding
from utils.dataloader import Dataset_A2D
from peft import LoraConfig
from transformers import TrainingArguments, logging, Trainer
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def train():
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

    train_dataset = Dataset_A2D(processor=processor,start_idx=0, end_idx=3500)
    eval_dataset = Dataset_A2D(processor=processor,start_idx=3500, end_idx=3600)

    peft_config = LoraConfig(
        target_modules=["q_proj", "k_proj"],
        init_lora_weights=False,
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.add_adapter(peft_config, adapter_name="Llasa_adapter")
    model.set_adapter('Llasa_adapter')

    for name, param in model.named_parameters():
        if "seg_projector" in name:
            if not torch.is_floating_point(param):
                param.data = param.data.float()  # Convert to float32 if not already
                # Alternatively, you can use double() or half() for float64 or float16
            param.requires_grad = True
        else:
            continue

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Trainable")

    torch.save(model.seg_projector.state_dict(), 'seg_projector_weights_before.pth')

    training_args = TrainingArguments(
        learning_rate = 2e-5,
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        output_dir="tmp",

        logging_strategy='steps',
        logging_steps=200,
        logging_first_step=True,
        
        save_strategy='epoch',

        disable_tqdm=False,
        num_train_epochs=5,
        log_level='error',
        report_to='wandb',
        fp16=True,

        eval_steps=1,
        evaluation_strategy='epoch'

    )

    logging.set_verbosity_error()

    trainer = Trainer(
        model=model, 
        args=training_args, 
        
        data_collator = LlasaDataCollatorWithPadding(processor=processor),
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
    )
    
    result = trainer.train()
    torch.save(model.seg_projector.state_dict(), 'seg_projector_weights_after.pth')

    print_summary(result)

if __name__ == '__main__':
    train()