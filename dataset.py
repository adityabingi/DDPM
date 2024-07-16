import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms.v2 as v2
from torchvision.datasets import CelebA

class CelebADataset(Dataset):
    
    def __init__(self, data_dir, split, img_size, hflip=False):
        
        self.ds = CelebA(root=data_dir, split=split, download=True)
        
        self.img_size = img_size
        transforms = [
                        v2.RandomHorizontalFlip(),
                        v2.ToTensor(),
                        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ]
        if not hflip:
            transforms = transforms[1:]
            
        self.transform = v2.Compose(transforms)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        image, _ = self.ds[idx]
        return self.transform(image)
    

def get_dataloader(data_dir, img_size, batch_size, num_workers, rank, world_size, split='train', distributed=False):
    
    hflip = True if split is 'train' else False 
    ds = CelebADataset(data_dir=data_dir, split=split, img_size=img_size, hflip=hflip)
    sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True,
    ) if distributed else None
    
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=num_workers,
    )
   
    
    return dl, sampler
    
    
