# IMG_1527.jpg
import os
import argparse
import logging
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

from models import SiT_models
from controlnet import ControlSiT
from download import find_model
from diffusers.models import AutoencoderKL
from transport import create_transport, Sampler
from train_utils import parse_transport_args
from maskdataset import PairedLayeredDataset, PairedTransform

# speed flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format=f'[\x1b[34m%%(asctime)s]\x1b[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f'{logging_dir}/log.txt')]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
            self.paths.extend(glob(os.path.join(root, ext)))
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img, edge = self.transform(img)
        else:
            img, edge = img, None
        
        return (img, edge), 0 
    
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
        experiment_name = f'{experiment_index:03d}-controlSiT-{model_string_name}-edges'
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
    from_pretrain = True  # args.from_pretrain

    # Load base checkpoint (required)
    if from_pretrain:
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
    
    if not from_pretrain:
        assert args.ckpt is not None, "Please provide --ckpt of the pretrained resume controlnet SiT."
        state_dict = find_model(args.ckpt)
        
        control_model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        
    requires_grad(ema, flag=False)
    
    if args.unfreeze_base:
        params = control_model.parameters()
    else:
        params =(p for p in control_model.parameters() if p.requires_grad)
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
    # transform = ImageWithCanny(args.image_size, args.canny_low, args.canny_high)
    transform = PairedTransform(args.image_size,is_training=args.is_training)

    if args.single_folder:
        dataset = FlatFolderDataset(args.data_path, transform=transform)
    else:
        # dataset = ImageFolder(args.data_path, transform=transform)
        dataset = PairedLayeredDataset(args.data_path, transform=transform)
    
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
    
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)
    
    control_model = DDP(control_model, device_ids=[rank])
    
    logger.info(f'Training for {args.epochs} epochs...')

    for epoch in range(args.epochs):
        
        ddp_sampler.set_epoch(epoch)
        
        logger.info(f'Beginning epoch {epoch}...')
        for (img, edge), y, _ in loader:
            img = img.to(device)
            edge = edge.to(device)
            y = y.to(device)
            
            if args.fixed_class_id >= 0:  # 将训练模型中的类别标签固定
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
                end_time = time()
                
                steps_per_sec = log_steps / (end_time - start_time)
                
                # All-reduce for average loss
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                logger.info(f'Train Steps: {train_steps:07d} Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}')
                
                running_loss = 0.0
                log_steps = 0
                start_time = time()
            
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": control_model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    ckpt_path = f'{ckpt_dir}/{train_steps:07d}.pt'
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")
                dist.barrier()
                
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples (with control)...")
                with torch.no_grad():
                    sample_fn = sampler.sample_ode()
                    ys = ys_sample
                    z=zs
                    if use_cfg:
                        z = torch.cat([z, z], dim=0)
                        y_null = torch.tensor([args.num_classes] * n, device=device)
                        ys = torch.cat([ys, y_null], dim=0)
                        model_fn = ema.forward_with_cfg
                        model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
                    else:
                        model_fn = ema.forward
                        model_kwargs = dict(y=ys)

                    
                    if args.no_sample_control:
                        ctrl_latents = None
                    else:
                        ctrl_latents = ctrl[0:len(z)]
                        
                    samples = sample_fn(z, model_fn, control=ctrl_latents, **model_kwargs)[-1]
                    dist.barrier()
                    
                    if use_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    
                    samples = vae.decode(samples / 0.18215).sample.clamp(-1, 1)
                    
                    img_vis = (img.float().detach().cpu()+1)* 0.5 # N, 3, H, W in [0, 1]
                    edge_vis = edge.float().detach().cpu() # N, 1, H, W in [0, 1]
                    edge_vis = (edge_vis+1)*0.5# 灰度图转3通道
                    
                    if edge_vis.shape[1] == 1:
                        edge_vis = edge_vis.repeat(1, 3, 1, 1)
                    
                    gen_vis = (samples.float().detach().cpu() + 1) * 0.5  # N, 3, H, W in [0, 1][0, 1]
                
                    # # 拼接展示: 原始图 + 控制图 + 生成图
                    n = len(img_vis) # n 已经定义
                    tiles = []
                    for i in range(n):
                        tiles.extend([img_vis[i], edge_vis[i], gen_vis[i]])
                        
                    grid = make_grid(tiles, nrow=3*int(np.sqrt(n)), padding=2) # # 这里的 nrow 有问题，应该是 3
                    
                    if dist.get_rank() == 0:

                        out_png = os.path.join(ckpt_dir, f'step_{train_steps:07d}.png')
                        save_image(grid, out_png)
                    
                logger.info("Sampling done.")
        
    control_model.eval()
    logger.info('Done!')
    cleanup()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000) # typically fewer epochs for control finetune
    
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["mse", "ema"], default="ema")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--cfg-scale", type=float, default=4.0)

   # Control-specific args
    parser.add_argument("--ckpt", type=str, required=True, help="Path to pretrained SiT checkpoint (.pt)")
    parser.add_argument("--canny-low", type=int, default=30)
    parser.add_argument("--canny-high", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--unfreeze-base", action="store_true", help="Whether to unfreeze the base SiT model for training.")
    parser.add_argument("--no-sample-control", action="store_true",help="Whether to disable control signals during sampling (for ablation).")
    parser.add_argument("--is-training", type=bool, help="Whether in training mode (enables data augmentation).") 
    # Single-class convenience
    parser.add_argument("--fixed-class-id", type=int, default=-1,help="If >=0, use this class ID for all training samples.")
    parser.add_argument("--single-folder", action="store_true", help="Whether to use a single folder of images instead of ImageFolder structure.")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)