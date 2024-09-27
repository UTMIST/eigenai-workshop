import torch
from diffusers import DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt

def create_noise_scheduler(config):
    return DDPMScheduler(num_train_timesteps=1000)

def plot_ddpm():
  T = 1000
  beta_start = 1e-4
  beta_end = 0.02
  betas = np.linspace(beta_start, beta_end, T)
  alphas = 1 - betas
  alphas_cumprod = np.cumprod(alphas)
  plt.figure(figsize=(10, 6))
  plt.subplot(2, 1, 2)
  plt.plot(alphas_cumprod, label='alpha cumulative product', linestyle='--')
  plt.title('Alpha and Cumulative Alpha Schedule')
  plt.xlabel('Timestep')
  plt.ylabel('Alpha Values')
  plt.legend()
  plt.tight_layout()
  plt.show()
