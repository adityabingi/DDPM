import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class Swish(nn.Module):
        
    def forward(x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    
    def __init__(self, d_model, max_len, hid_dim):
        assert d_model % 2 == 0
        super().__init__()
        
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(max_len).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [max_len, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [max_len, d_model // 2, 2]
        emb = emb.view(max_len, d_model)
        
        self.time_embedding = nn.Sequential(
                                 nn.embedding.from_pretrained(emb),
                                 nn.Linear(d_model, hid_dim),
                                 Swish(),
                                 nn.Linear(hid_dim, hid_dim)
                                 )
        
    def forward(self, t):
        
        emb = self.time_embedding(t)
        return emb
    
    
class DownSample(nn.Module):
    
    def __init__(self, n_ch):
        super().__init__()
        
        self.conv =  nn.Conv2d(in_channels = n_ch, out_channels = n_ch, kernel_size =3, stride=2, padding=1)  
    
    def forward(self,x):
        x = self.conv(x)
        return x

class UpSample(nn.Module):
    
    def __init__(self, n_ch):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels = n_ch, out_channels = n_ch, kernel_size =3, stride=1, padding=1)
    
    def forward(self, x):
        
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        
        return x
    
class AttnBlock(nn.Module):
    
    def __init__(self, in_ch):
        super().__init__()
        
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        
    def forward(self, x):
        
        b, c, h, w = x.shape
        h = self.group_norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        q = q.permute(0, 2,3, 1).view(b, h*w, c)
        k = k.view(b, c, h*w)
        attn_w = torch.bmm(q, k) * (int(c)**(-0.5))
        assert w.shape == (b, h*w, h*w)
        w = F.softmax(w, dim=-1)
        
        v = v.permute(0, 2, 3, 1).view(b, h*w, c)
        h = torch.bmm(attn_w, v)
        assert x.shape == (b, h*w, c)
        h = h.view(b, h, w, c).permute(0, 3, 1, 2)
        
        h = self.proj(h)
        
        h = x+h
        
        return h  
    
class ResBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        
        self.block1 = nn.Sequential(
                             nn.GroupNorm(32, in_ch),
                             Swish(),
                             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0))
        
        self.temb_proj = nn.Sequential(
                             Swish(),
                             nn.Linear(tdim, out_ch))
        
        self.block2 = nn.Sequential(
                             nn.GroupNorm(32, out_ch),
                             Swish(),
                             nn.Dropout(dropout),
                             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0)
        )
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock()
        else:
            self.attn = nn.Identity()     
    
    
    def forward(self, x, temb):
        
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h) 
        return h
    
class UNet(nn.Module):
    
    def __init__(self, ch, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=2, dropout=0.1):
        super().__init__()
        
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(max_len=1000, d_model=ch, dim=tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        cxs = [ch]  # record output channel when dowmsample for upsample
        cur_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=cur_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn)
                    )
                )
                cur_ch = out_ch
                cxs.append(cur_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(cur_ch))
                cxs.append(cur_ch)
                
        self.middleblocks = nn.ModuleList([
            ResBlock(cur_ch, cur_ch, tdim, dropout, attn=True),
            ResBlock(cur_ch, cur_ch, tdim, dropout, attn=False),
        ])
        
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=cxs.pop() + cur_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                cur_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(cur_ch))
                
        assert len(cxs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
            Swish(),
            nn.Conv2d(cur_ch, 3, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, noisy_image, diffusion_step):
        
        temb = self.time_embedding(diffusion_step)
        x = self.head(noisy_image)
        xs = [x]
        for layer in self.downblocks:
            if isinstance(layer, DownSample):
                x = layer(x)
            else:
                x = layer(x, temb)
            xs.append(x)

        for layer in self.middleblocks:
            x = layer(x, temb)

        for layer in self.upblocks:
            if isinstance(layer, UpSample):
                x = layer(x)
            else:
                x = torch.cat([x, xs.pop()], dim=1)
                x = layer(x, temb)
        x = self.tail(x)
        assert len(xs) == 0
        return x
