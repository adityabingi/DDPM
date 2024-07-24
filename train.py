import torch
from torch.optim import AdamW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import math
import time
#from timm.scheduler import CosineLRScheduler
import wandb
import argparse

from config import Config
from dataset import get_dataloader
from utils import set_seed, image_to_grid, save_image, get_elapsed_time
from unet import UNet
from ddpm import DDPM

from pathlib import Path

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True



class Trainer:
    
    def __init__(self, train_dl, val_dl, model, optimizer, rank, device, run):
        
        self.model = model
        self.optim = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.rank = rank
        self.device = device
        self.run = run
        
        self.ckpt_path = Path(Config.ckpt_dir + f"{self.run.name}/ckpt.pth")
        self.log_dir = Config.log_dir + f"{self.run.name}/"
    
    
    def train_one_epoch(self, epoch):
        
        self.train_dl.sampler.set_epoch(epoch)
        train_loss = 0
        for img_batch in self.train_dl:
            
            img_batch = img_batch.to(self.device)
            
            self.optim.zero_grad()
            loss = self.model(img_batch)
            
            train_loss += (loss.item() / len(self.train_dl))
            
            loss.backward()
            self.optim.step()
            
        return train_loss
     
    @torch.inference_mode        
    def validate(self):
        
        val_loss = 0
        
        for img_batch in self.val_dl:
            img_batch = img_batch.to(self.device)
            
            loss = self.model(img_batch)
            
            val_loss += (loss.item()/len(self.val_dl))
        
        return val_loss
    
    def train(self):
        
        start_time = time.time()
        self.model = torch.compile(self.model)
        
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, len(self.train_dl), )
        min_val_loss = -math.inf
        for epoch in range(1, Config.num_epochs+1):
            
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            
            if val_loss < min_val_loss and self.rank==0:
                min_val_loss = val_loss
                
            self.save_ckpt(min_val_loss, epoch)
                    
            self.test_samples(epoch=epoch, test_batch_size=16)
                    
            
            if self.rank == 0:
                
                log = f"[ {get_elapsed_time(start_time)} ]"
                log += f"[ {epoch}/{Config.num_epochs} ]"
                log += f"[ Train loss: {train_loss:.4f} ]"
                log += f"[ Val loss: {val_loss:.4f} | Best: {min_val_loss:.4f} ]"
                print(log)
                wandb.log(
                    {"Train loss": train_loss, "Val loss": val_loss, "Min val loss": min_val_loss},
                    step=epoch,
                )
                
        self.run.finish()  
              
    @torch.inference_mode    
    def test_samples(self, epoch, test_batch_size):
        
        if self.rank == 0:
            gen_image = self.model.module.sample(shape=[test_batch_size,] + Config.img_size)
            gen_grid = image_to_grid(gen_image, n_cols=int(test_batch_size ** 0.5))
            sample_path = str(
                self.log_dir/self.run.name/f"sample_imgs/sample-epoch={epoch}.jpg"
            )
            save_image(gen_grid, save_path=sample_path)
            wandb.log({"Samples": wandb.Image(sample_path)}, step=epoch)
   
    
    def save_ckpt(self, min_val_loss, epoch):
        if self.rank ==0:
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "min_val_loss": min_val_loss,
                }
                
            torch.save(ckpt, str(self.ckpt_path))
        

def main_worker(rank, world_size, run):
    
    setup(rank, world_size, Config.port)
    
    DEVICE = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device('cpu')
    set_seed(Config.seed + rank)
    print(f"[ DEVICE: {DEVICE} ][ RANK: {rank} ]")
    
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
    assert Config.batch_size % world_size==0
    batch_size_per_gpu = Config.batch_size //world_size
        
    train_dl = get_dataloader(Config.data_dir, Config.img_size, batch_size_per_gpu, Config.num_workers, rank, world_size, distributed=True)
    val_dl = get_dataloader(Config.data_dir, Config.img_size, batch_size_per_gpu, Config.num_workers,  rank, world_size, distributed=True)
    
    unet = UNet().to(DEVICE)
    ddpm = DDPM(unet, device=DEVICE).to(DEVICE)
    
    if "cuda" in str(DEVICE):
        ddpm = DDP(ddpm, device_ids=[rank])
    else:
        ddpm = DDP(ddpm)
    optim = AdamW(ddpm.parameters(), lr=Config.lr)
    
    trainer = Trainer(train_dl, val_dl, ddpm, optim, rank, DEVICE, run)
    dist.barrier()
    trainer.train()
    
    cleanup()
    
def setup(rank, world_size, port):

    # initialize the process group
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main():
    
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--train", type=str, required=True)
    # parser.add_argument("--sample", type=str, required=True)
    # parser.add_argument("--test", type=int, required=True)
    
    # args = parser.parse_args()
    run = wandb.init(project="DDPM_prototype")
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size=1
    mp.spawn(
            main_worker,
            args=(world_size, run),
            nprocs=world_size,
            join=True,
        )
    

if __name__ == '__main__':
    main()  

