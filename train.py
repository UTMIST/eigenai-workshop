import os
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from model import create_diffusion_model
from data import create_dataloader
from scheduler import create_noise_scheduler
from inference import denoising_process
from utils import forward_diffusion, create_accelerator, create_optimizer_and_lr_scheduler, create_default_config
from tqdm import tqdm
from diffusers import DDPMPipeline
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid



def evaluate(config, model,scheduler,epoch):
    denoising_process(config,model,scheduler)
    return 

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler,accelerator):
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            timesteps = torch.randint(0, 1000, (config.train_batch_size,), device=clean_images.device,dtype=torch.int64)            
            noisy_images = forward_diffusion(scheduler=noise_scheduler,clean_images=clean_images,timesteps=timesteps,noise=noise)
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, accelerator.unwrap_model(model),noise_scheduler,epoch)
            if not os.path.isdir(f"{config.output_dir}/checkpoints"):
                os.mkdir(f"{config.output_dir}/checkpoints")
            torch.save(accelerator.unwrap_model(model).state_dict(), f"{config.output_dir}/checkpoints/checkpoint-{epoch}.pt") 

if __name__ == "__main__":
    config = create_default_config()
    dataloader = create_dataloader(config)
    model = create_diffusion_model(config)
    noise_scheduler= create_noise_scheduler(config)
    accelerator = create_accelerator(config)
    optimizer,lr_scheduler= create_optimizer_and_lr_scheduler(config,model)
    train_loop(config, model, noise_scheduler, optimizer, dataloader, lr_scheduler,accelerator)