import torch
from diffusers import DDPMScheduler

def create_noise_scheduler(config):
    return DDPMScheduler(num_train_timesteps=1000)

class DDPM_scheduler:
  def __init__(self,diffusion_timesteps):
    self.diffusion_timesteps = diffusion_timesteps
    self.inference_timesteps = diffusion_timesteps
    self.training_timesteps = diffusion_timesteps
    self.beta =  torch.linspace(0.0001,0.02,1000)
    self.alpha = 1 - self.beta
    self.alphas_bar = torch.cumprod(self.alpha,0)
    self.one_minus_alphas_bar = 1 - self.alphas_bar

  def step(
      self,
      pred_noise:torch.Tensor, # output from the noise-prediction model (U-Net)
      t: int, # the current denoising timestep
      x_t:torch.Tensor, # current sample
  ):
    '''
    '''

    # Step 1: Get the previous timestep (in DDPM, inference timesteps = training timesteps so t_prev = t -1)
    t_prev = t - self.inference_timesteps // self.training_timesteps

    # Step 2: compute all the necessary noise schedule constants (alphas, betas, alpha prods, beta prods)
    alpha_t = self.alpha[t]
    beta_t = self.beta[t]
    alpha_prod_t = self.alphas_bar[t]
    alpha_prod_t_prev = self.alphas_bar[t-1] if t >=0 else torch.ones
    beta_prod_t = 1 - self.alphas_bar[t]

    # Step 3: compute pred_x0, that is needed in equation 15 of DDPM paper
    pred_x0 = (x_t - beta_prod_t ** -0.5 * pred_noise) / alpha_prod_t **0.5
    #NOTE: in this step we are essentially rearranging for x0 from the forward
    # diffusion formula: x0 * alpha_prod_t **0.5 + beta_prod_t ** -0.5 * noise = x_t
    #                    |-----signal-----------|  |-------noise--------------|


    # Step 4: Compute one backward step (Langevin Dynamic Sampling),
    #         correspond to line 4 in Algorithm 2 of DDPM paper
    pred_x0_coeff = (alpha_prod_t_prev ** (0.5) * beta_t) / beta_prod_t
    xt_coeff = alpha_t ** 0.5 * beta_t


    # Step 5: get the variance we want to add to the model
    variance = torch.randn(pred_noise.shape)
    pred_xt_prev = pred_x0_coeff * pred_x0 + xt_coeff * x_t
    pred_xt_prev = pred_xt_prev + variance
    return pred_xt_prev


if __name__ == "__main__":
   torch.manual_seed(10)
   ddpm = DDPMScheduler(num_train_timesteps=1000)
   ddpm.set_timesteps(1000)
   my_ddpm = DDPM_scheduler(1000)
   x = torch.randn(16,3,28,28)
   model_output = torch.randn(16,3,28,28)
   sample = torch.randn(16,3,28,28)

   timestep = 500
   import ipdb; ipdb.set_trace()
   out_1 = ddpm.step(model_output = model_output,timestep= timestep,sample=sample)
   out_2 = my_ddpm.step(model_output,timestep,sample)
