
import os
import random
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler
from datetime import timedelta
from time import time
from PIL import Image
from pathlib import Path
from collections import OrderedDict

from torchvision.utils import make_grid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def create_dir(x):
    x = Path(x)
    if x.is_dir():
        x.mkdir(parents=True, exist_ok=True)
    else:
        x.parent.mkdir(parents=True, exist_ok=True)


def save_image(image, save_path):
    create_dir(save_path)
    _to_pil(image).save(str(save_path), quality=100)


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def denorm(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    return TF.normalize(
        x, mean=-(np.array(mean) / np.array(std)), std=(1 / np.array(std)),
    )


def image_to_grid(image, n_cols):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid

