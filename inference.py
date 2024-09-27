import os
import torch
import diffusers
from diffusers.utils import make_image_grid
import torchvision
from PIL import Image

def denoising_process(config,model,scheduler,save_images = True,image_name="sample"):
    with torch.no_grad():
        image = torch.randn(config.train_batch_size,model.config.in_channels, config.image_size, config.image_size).to("cuda")
        for t in scheduler.timesteps:
            model_output = model(image,t).sample
            image = scheduler.step(model_output,t,image).prev_sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        image_grid = make_image_grid(pil_images, rows=4, cols=4)
        test_dir = os.path.join(config.output_dir, "samples")
        image_grid.save(f"{test_dir}/{image_name}.png")
        return



