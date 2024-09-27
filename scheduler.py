import torch
from diffusers import DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt

def create_noise_scheduler(config):
    return DDPMScheduler(num_train_timesteps=1000)

def create_ddpm_plot(steps, beta_schedule, alpha_schedule, alpha_cumprod_schedule):
    """
    Helper function to handle the plotting logic for DDPM schedules.
    """
    fig, ax1 = plt.subplots()
    ax1.plot(steps, beta_schedule, 'b-', label='Beta Schedule', linewidth=2)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Beta Schedule', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.plot(steps, alpha_schedule, 'g-', label='Alpha Schedule', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(steps, alpha_cumprod_schedule, 'r-', label='Cumulative Alpha Schedule', linewidth=2)
    ax2.set_ylabel('Cumulative Alpha Schedule', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('DDPM Beta, Alpha, and Cumulative Alpha Schedules')
    ax1.grid()

def plot_ddpm():
  # Simulated data for DDPM schedules
  T = 1000  # Total steps
  steps = np.arange(1, T + 1)

  # Beta schedule (linear)
  beta_schedule = np.linspace(0.0001, 0.02, T)

  # Alpha schedule
  alpha_schedule = 1 - beta_schedule

  # Cumulative alpha schedule
  alpha_cumprod_schedule = np.cumprod(alpha_schedule)

  # Call the plot helper function
  create_ddpm_plot(steps, beta_schedule, alpha_schedule, alpha_cumprod_schedule)