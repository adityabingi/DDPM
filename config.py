class Config:
    
    # training params 
    lr = 0.0002
    num_epochs = 1000
    batch_size = 128
    lr_schedule = "cosine"
    warmup_steps = ""
    
    # Data Params
    data_dir = "celeba/"
    img_size = [3, 64, 64]    # [C, H, W]
    
    # data loader worker threads
    num_workers = 8

    # diffusion params
    diffusion_timesteps = 1000
    
    
    # multi-gpu params
    port = 8888
    
    ckpt_dir = "ckpts/"
    log_dir = "logs/"
    
    # random seed
    seed = 42
    
