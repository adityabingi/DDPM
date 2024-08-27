import torch
import argparse

from config import Config
from utils import image_to_grid, save_image, modify_state_dict
from unet import UNet
from ddpm import DDPM


@torch.inference_mode  
def sample(model, sample_batch_size, sampler, run_name):
    
    gen_image = model.sample(shape=[sample_batch_size,] + Config.img_size, sampler=sampler)
    gen_grid = image_to_grid(gen_image, n_cols=int(sample_batch_size ** 0.5))
    sample_path = Config.log_dir + run_name + f"/inference/{sampler}_sample.jpg"
    save_image(gen_grid, save_path=sample_path)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    args = parser.parse_args()

    DEVICE = torch.device(f"cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    cp = torch.load(args.ckpt_path)
    run_name = args.ckpt_path.split("/")[-2]
    unet =UNet().to(DEVICE)
    ddpm = DDPM(unet, device=DEVICE)
    ddpm.load_state_dict(modify_state_dict(cp['model']))
    ddpm.eval()
    sample(ddpm, args.sample_batch_size, args.sampler, run_name)
    
if __name__ == '__main__':
    main()  
