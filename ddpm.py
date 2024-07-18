import torch
import torch.nn as nn
from torch.nn import functional as F

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class DDPM(nn.Module):
    
    def __init__(self, model, device, beta_start, beta_end, timesteps=1000):
        super().__init__()
        
        self.model = model
        self.device = device
        self.num_timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        self.register_buffer('betas',  self.get_linear_schedule())
        self.alphas = 1 - self.betas 
        
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        self.register_buffer('sqrt_oneminus_alphas_bar', torch.sqrt(1-self.alphas_bar))
        
        self.register_buffer('coeff1', 1/torch.sqrt(self.alphas))
        self.register_buffer('coeff2', self.coeff1 * ((1 - self.alphas)/(self.sqrt_oneminus_alphas_bar)))
    
    
    def get_linear_schedule(self):
        
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).double().to(self.device)
    
    def q_sample(self, x_0, t):
        
        noise = torch.randn_like(x_0)
        
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + extract(self.sqrt_oneminus_alphas_bar, t, x_0.shape) * noise
        
        return x_t, noise
        
    def forward(self, x_0):
        
        t = torch.randint(self.num_timesteps, size=(x_0.shape[0], ), device=self.device)
        x_t, noise = self.q_sample(x_0, t)
        noise_pred = self.model(x_t, t)
        loss = F.mse(noise_pred, noise)
        return loss
    
    @torch.inference_mode
    def p_sample(self, x_t, t):
        
        eps = torch.randn_like(x_t)
        eps_theta = self.model(x_t, t)
        mean = extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps_theta
        
        sample = mean + torch.sqrt(self.betas) * eps
        
        return sample

