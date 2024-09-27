from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from dataclasses import dataclass
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
import os

@dataclass(repr=True)
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "utmist-workshop-diffusion"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


def create_accelerator(config):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("utmist-workshop-train")
    return accelerator


def forward_diffusion(scheduler,clean_images,timesteps,noise):
        noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
        return noisy_images

def create_optimizer_and_lr_scheduler(config,model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.training_steps,
    )
    return optimizer,lr_scheduler

def create_default_config():
     return TrainingConfig()


if __name__ == "__main__":
     config = create_default_config()
     print(config)