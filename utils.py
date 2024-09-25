from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
import os

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
