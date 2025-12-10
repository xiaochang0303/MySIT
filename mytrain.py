# IMG_1527.jpg
import os
import argparse
import logging
import glob
import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler

from torchvision.datasets import ImageFolder
from torchvision import transforms

from models import SiT_models
from controlnet import ControlSiT
from loadmodel import find_model
from diffusers.models import AutoencoderKL
from transport import create_transport, Sampler
from train_utils import parse_transport_args

# speed flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad() # 2 usages
def update_ema(ema_model, model, decay=0.99):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
# IMG_1517.JPG
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    # 1 usage
    dist.destroy_process_group()

def create_logger(logging_dir):
    # 2 usages
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format=f'[%(levelname)s %(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f'{logging_dir}/log.txt')]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size, crop_ratio=1.0):

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)

    # # 这里把原来的 image_size 变成了 crop_size
    if isinstance(crop_ratio, float):
        crop_ratio = crop_ratio
    else:
        crop_ratio = np.random.uniform(crop_ratio[0], crop_ratio[1])
    crop_size = int(round(image_size * crop_ratio)) # 必须是整数
    crop_y = (arr.shape[0] - crop_size) // 2
    crop_x = (arr.shape[1] - crop_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + crop_size, crop_x: crop_x + crop_size])

class ImageWithCanny(transforms.Compose):
    # 1 usage
    def __init__(self, image_size, low=100, high=200):
        self.image_size = image_size
        self.low = low
        self.high = high
        super().__init__([
            transforms.Lambda(lambda pil: center_crop_arr(pil, image_size, crop_ratio=1.0)),
            transforms.RandomHorizontalFlip(),
        ])
    
    def __call__(self, img):
        # Apply base transforms (crop/flip) deterministically to both views
        img_t = super().__call__(img)
        # Canny on grayscale numpy
        np_img = np.array(img)
        
        if np_img.ndim == 3:
            gray = (0.299 * np_img[..., 0] + 0.587 * np_img[..., 1] + 0.114 * np_img[..., 2]).astype(np.uint8)
        else:
            gray = np_img.astype(np.uint8)        
        try:
            import cv2
            edges = cv2.Canny(gray, self.low, self.high)
        except Exception:
            # Fallback: simple Sobel magnitude threshold
            gy = np.zeros_like(gray, dtype=np.float32)
            gx = np.zeros_like(gray, dtype=np.float32)
            gy[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            gx[1:-1, :] = gray[2:, :] - gray[:-2, :]
            
            # Simple thresholding
            edges = (np.hypot(gx, gy) > 64).astype(np.uint8) * 255

        img_t = transforms.ToTensor()(img) # [0, 1]
        img_t = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_t) # [-1, 1]
        
        edge_t = torch.from_numpy(edges).float().unsqueeze(0) / 255.0 # [1, H, W]
        
        return img_t, edge_t

# IMG_1519.JPG
class FlatFolderDataset(Dataset):
    # 1 usage
    def __init__(self, root, transform=None):
        self.paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
            self.paths.extend(glob(os.path.join(root, ext)))
        self.transform = transform
    
    def __len__(self):
        # 0 usages
        return len(self.paths)
        
    def __getitem__(self, idx):
        # 0 usages
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img, edge = self.transform(img)
        else:
            img, edge = img, None
        
        return (img, edge), 0 # img, edge, # y=0 (will be overridden if --fixed-class-id=0)

# IMG_1519.JPG
def main(args):
    assert torch.cuda.is_available(), 'Training currently requires at least one GPU.'

    # # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_bs = args.global_batch_size // dist.get_world_size()

    # # Folders & logger
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f'{args.results_dir}/*'))
        model_string_name = args.model.replace('/', '-')
        experiment_name = f'{experiment_index:030}-controlSiT-{model_string_name}-edges'
        experiment_dir = f'{args.results_dir}/{experiment_name}'
        ckpt_dir = f'{experiment_dir}/checkpoints'
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f'Experiment dir: {experiment_dir}')

    else:
        logger = create_logger(None)


    # Model backbone
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    base = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    from_ptrttrain = True # args.from_prttrain

    # Load base checkpoint (required)
    if from_ptrttrain:
        assert args.ckpt is not None, "Please provide --ckpt of the pretrained SiT to build ControlNet on."
        state_dict = find_model(args.ckpt)
        
        del state_dict['ema']['y_embedder.embedding_table.weight']
        missing, unexpected = base.load_state_dict(state_dict["ema"], strict=False)
        
        if len(missing) > 0:
            print(f"[INFO] missing keys (usually fine if you froze base during train): {len(missing)}")
            print(missing)
        if len(unexpected) > 0:
            print(f"[INFO] unexpected keys: {len(unexpected)}")
            print(unexpected)

    control_model = ControlSiT(base, freeze_base=not args.unfreeze_base).to(device)
    ema = deepcopy(control_model).to(device)
    
    if not from_ptrtrain:
        assert args.ckpt is not None, "Please provide --ckpt of the pretrained resume controlnet SiT."
        state_dict = find_model(args.ckpt)
        
        control_model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        
    requires_grad(ema, flag=False)
    
    if args.unfreeze_base:
        params = control_model.parameters()
    else:
        params = [p for p in control_model.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    
    # # Transport / sampler
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    
    vae = AutoencoderKL.from_pretrained(f'./sd-vae-ft-{args.vae}').to(device)
    vae.eval()
    
    # Set SiT Parameters (trainable):
    logger.info(f"SiT Parameters (trainable): {sum(p.numel() for p in control_model.parameters() if p.requires_grad):,}")
    
    # Data
    transform = ImageWithCanny(args.image_size, args.canny_low, args.canny_high)
    if args.single_folder:
        dataset = FlatFolderDataset(args.data_path, transform=transform)
    else:
        dataset = ImageFolder(args.data_path, transform=transform)
    
    ddp_sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset, batch_size=local_bs, shuffle=False, sampler=ddp_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )
    logger.info(f'Dataset contains {len(dataset)} images ({args.data_path}).')
    
    update_ema(ema, control_model, decay=0) # copy
    control_model.train()
    ema.eval()
    
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    
    use_cfg = args.cfg_scale > 1.0
    n = local_bs
    
    if args.fixed_class_id >= 0:
        ys_sample = torch.full((n,), args.fixed_class_id, device=device, dtype=torch.long)
    else:
        ys_sample = torch.randint(args.num_classes, size=(n,), device=device)
    
    zs = torch.randn(n, latent_size, latent_size, device=device)
    
    control_model = DDP(control_model, device_ids=[rank])
    
    logger.info(f'Training for {args.epochs} epochs...')

    for epoch in range(args.epochs):
        
        ddp_sampler.set_epoch(epoch)
        
        logger.info(f'Beginning epoch {epoch}...')
        for (img, edge), y in loader:
            img = img.to(device) # [-1, 1]
            edge = edge.to(device) # [0, 1] shape [N, 1, H, W]
            y = y.to(device)
            
            if args.fixed_class_id >= 0: # # 将训练模型中的类别标签固定
                y = torch.full_like(y, args.fixed_class_id)
                
            with torch.no_grad():
                # Encode images to latents
                x = vae.encode(img).latent_dist.sample().mul_(0.18215)
                
                # Turn edge into 1ch in [-1, 1] then encode to latents for control
                if edge.shape[1] == 1:
                    edge_rgb = edge.repeat(1, 3, 1, 1)
                else:
                    edge_rgb = edge
                    
                ctrl = vae.encode(edge_rgb).latent_dist.sample().mul_(0.18215)
                
            model_kwargs = dict(y=y, control=ctrl)
            
            loss_dict = transport.training_losses(control_model, x, model_kwargs)
            loss = loss_dict['loss'].mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            update_ema(ema, control_model.module)
            

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                
                steps_per_sec = log_steps / (end_time - start_time)
                
                # All-reduce for average loss
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                logger.info(f'Train Steps: {train_steps:07d} Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}')
                
                running_loss = 0.0
                log_steps = 0
                start_time = time.time()
            
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": control_model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    ckpt_path = f'{ckpt_dir}/train_steps:{train_steps:07d}.pt'
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")
                dist.barrier()
                
            if train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info("Generating EMA samples (with control)...")
                    with torch.no_grad():
                        sampler_ode = sampler.sample_ode(
                            ys_sample,
                            z=zs,
                            use_cfg=use_cfg,
                            y_null = torch.tensor([args.num_classes] * n, dim=0, device=device)
                        )