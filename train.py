import os
import torch
import torchvision
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision import transforms
from model import create_diffusion_model
from data import create_dataloader
from scheduler import create_noise_scheduler
from utils import forward_diffusion, create_accelerator, create_optimizer_and_lr_scheduler
from tqdm import tqdm
from diffusers import DDPMPipeline
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
@dataclass
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


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler,accelerator):
    # Initialize accelerator and tensorboard logging

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            timesteps = torch.randint(0, 1000, (config.train_batch_size,), device=clean_images.device,dtype=torch.int64)            
            noisy_images = forward_diffusion(scheduler=noise_scheduler,clean_images=clean_images,timesteps=timesteps,noise=noise)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

if __name__ == "__main__":
    config = TrainingConfig()
    dataloader = create_dataloader(config)
    model = create_diffusion_model(config)
    noise_scheduler= create_noise_scheduler(config)
    accelerator = create_accelerator(config)
    optimizer,lr_scheduler= create_optimizer_and_lr_scheduler(config,model)
    train_loop(config, model, noise_scheduler, optimizer, dataloader, lr_scheduler,accelerator)