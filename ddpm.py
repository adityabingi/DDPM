import torch
import torch.nn as nn
from torch.nn import functional as F

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).contiguous().float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class DDPM(nn.Module):
    
    def __init__(self, 
                 model, 
                 device, 
                 beta_start=0.0001, 
                 beta_end=0.02, 
                 timesteps=1000,):
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
        
        noise = torch.randn_like(x_0).to(self.device)
        
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + extract(self.sqrt_oneminus_alphas_bar, t, x_0.shape) * noise
        
        return x_t, noise
        
    def forward(self, x_0):
        
        t = torch.randint(self.num_timesteps, size=(x_0.shape[0], ), device=self.device)
        x_t, noise = self.q_sample(x_0, t)
        noise_pred = self.model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    @torch.inference_mode
    def p_sample(self, x_t, t):
        
        noise = torch.randn_like(x_t) if t>0 else torch.zeros_like(x_t)
        noise = noise.to(self.device)
        
        t = torch.full(
            size=(x_t.shape[0],),
            fill_value=t,
            dtype=torch.long,
            device=self.device,
        )   
        eps_theta = self.model(x_t, t)
        mean = extract(self.coeff1, t, x_t.shape) * (x_t - extract(self.coeff2, t, x_t.shape) * eps_theta)
        
        sample = mean + extract(torch.sqrt(self.betas), t, x_t.shape) * noise
        
        return sample
    
    @torch.inference_mode
    def sample(self, shape, sampler="ddpm", ddim_steps=20, ddim_eta=0.0):
        
        if sampler=="ddim":
            x_t = self.ddim_sample(shape, ddim_steps, ddim_eta)
            return x_t
        
        x_t = torch.randn(shape).to(self.device)
        for t in reversed(range(self.num_timesteps)):
            x_t = self.p_sample(x_t, t)
            
        return x_t
    
    @torch.inference_mode
    def ddim_sample(self, shape, ddim_steps, ddim_eta):
        
        skip_steps = self.num_timesteps // ddim_steps
        seq = list(range(0, self.num_timesteps, skip_steps))         
        seq_prev = [-1] + seq[:-1]           
        x_t = torch.randn(shape).to(self.device)             
                                                                     
        for t, t_prev in zip(reversed(seq), reversed(seq_prev)):
            if t_prev<0:
                break
            
            noise = torch.randn_like(x_t) if t>0 else torch.zeros_like(x_t)
            noise = noise.to(self.device)
            
            t = torch.full(
                size=(x_t.shape[0],),
                fill_value=t,
                dtype=torch.long,
                device=self.device,)   
            
            t_prev = torch.full(
                size=(x_t.shape[0],),
                fill_value=t_prev,
                dtype=torch.long,
                device=self.device,)   
            
            eps_theta = self.model(x_t, t)
            
            alpha = extract(self.alphas_bar, t, x_t.shape)
            alpha_prev = extract(self.alphas_bar, t_prev, x_t.shape)
            
            sigma = ddim_eta * ((1-alpha/alpha_prev)*((1-alpha_prev)/(1-alpha))).sqrt()
            c = (1-alpha_prev-(sigma**2)).sqrt()
            x_start = (x_t - ((1-alpha).sqrt() * eps_theta))/alpha.sqrt()
            x_t = alpha_prev.sqrt() * x_start + c * eps_theta + sigma * noise
            
        return x_t

