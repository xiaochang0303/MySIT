import os
import argparse
import logging
import glob
import time
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from models import SiT_models
from controlnet_sit import ControlSiT
from diffusers.models import AutoencoderKL
from transport import create_transport, Sampler
from train_utils import parse_transport_args
from maskdataset import PairedTransform, PairedLayeredDataset, ImageWithCanny, center_crop_arr
# speed flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
            self.paths.extend(glob.glob(os.path.join(root, ext)))
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
        
        # 返回文件名（不包含路径和扩展名）
        filename = os.path.splitext(os.path.basename(path))[0]
        return (img, edge), 0, filename

class ImageFolderWithFilename(ImageFolder):
    """ImageFolder的包装类，返回文件名"""
    def __getitem__(self, idx):
        (img, edge), label = super().__getitem__(idx)
        # 获取文件路径
        path, _ = self.samples[idx]
        filename = os.path.splitext(os.path.basename(path))[0]
        return (img, edge), label, filename

def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()

def main(args):
    # 初始化分布式训练
    use_distributed = args.num_gpus > 1
    
    if use_distributed:
        assert torch.cuda.is_available(), 'Multi-GPU generation requires CUDA.'
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        print(f"Rank {rank}/{world_size} using GPU {device}")
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    
    torch.manual_seed(args.global_seed)
    
    if rank == 0:
        print(f"Starting generation with seed={args.global_seed}.")
        print(f"Using {world_size} GPU(s) for generation.")

    # Setup folders
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)

    # Model backbone
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    base = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)

    # Initialize ControlNet
    control_model = ControlSiT(base, freeze_base=True).to(device)
    control_model.eval()

    # Load Checkpoint
    if rank == 0:
        print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    if "ema" in checkpoint:
        control_model.load_state_dict(checkpoint["ema"])
    elif "model" in checkpoint:
        control_model.load_state_dict(checkpoint["model"])
    else:
        control_model.load_state_dict(checkpoint)

    # Transport / Sampler
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    
    # VAE
    if rank == 0:
        print(f"Loading VAE: {args.vae}")
    vae = AutoencoderKL.from_pretrained(f'./sd-vae-ft-{args.vae}').to(device)
    vae.eval()
    
    # Data - 根据 control_type 选择不同的 transform 和 dataset
    if args.control_type == 'mask':
        transform = PairedTransform(args.image_size, is_training=False)
        if args.single_folder:
            dataset = FlatFolderDataset(args.data_path, transform=transform)
        else:
            dataset = PairedLayeredDataset(args.data_path, transform=transform)
    
    if args.control_type == 'canny':
        transform = ImageWithCanny(args.image_size, args.canny_low, args.canny_high, is_training=False)
        if args.single_folder:
            dataset = FlatFolderDataset(args.data_path, transform=transform)
        else:
            dataset = ImageFolderWithFilename(args.data_path, transform=transform)
    
    # if args.control_type == 'sobel':
    #     # TODO: 实现 sobel transform 和相应的 dataset
    #     pass
    
    # 使用分布式采样器
    if use_distributed:
        sampler_data = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler_data = None
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False if use_distributed else False,
        sampler=sampler_data,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    if rank == 0:
        print(f"Dataset contains {len(dataset)} images.")
        print(f"Each GPU will process ~{len(dataset) // world_size} images.")
    
    cnt = 0
    start_time = time.time()
    
    for i, batch_data in enumerate(loader):
        # 处理不同数据集的返回格式
        if len(batch_data) == 3:
            (img, edge), y, filenames = batch_data
        else:
            (img, edge), y = batch_data
            filenames = [f"sample_{cnt + j:05d}" for j in range(len(y))]
        
        img = img.to(device)
        edge = edge.to(device)
        y = y.to(device)
        
        n = img.shape[0]
        
        # Override class id if specified
        if args.fixed_class_id >= 0:
            y = torch.full_like(y, args.fixed_class_id)
            
        with torch.no_grad():
            # 1. Encode Control Signal (Edge)
            if edge.shape[1] == 1:
                edge_rgb = edge.repeat(1, 3, 1, 1)
            else:
                edge_rgb = edge
            
            # Encode edge to latent
            ctrl = vae.encode(edge_rgb).latent_dist.sample().mul_(0.18215)
            
            # 2. Prepare Latents z
            z = torch.randn(n, latent_size, latent_size, device=device)
            
            # 3. Setup Classifier-Free Guidance
            use_cfg = args.cfg_scale > 1.0
            if use_cfg:
                z_in = torch.cat([z, z], dim=0)
                y_null = torch.tensor([args.num_classes] * n, device=device)
                y_in = torch.cat([y, y_null], dim=0)
                
                # Handle control for CFG
                if args.no_sample_control:
                    ctrl_in = None
                else:
                    ctrl_in = torch.cat([ctrl, ctrl], dim=0)
                
                model_kwargs = dict(y=y_in, cfg_scale=args.cfg_scale)
                model_fn = control_model.forward_with_cfg
            else:
                z_in = z
                y_in = y
                
                if args.no_sample_control:
                    ctrl_in = None
                else:
                    ctrl_in = ctrl
                    
                model_kwargs = dict(y=y_in)
                model_fn = control_model.forward

            # 4. Sample
            sample_fn = sampler.sample_ode()
            samples = sample_fn(z_in, model_fn, control=ctrl_in, **model_kwargs)[-1]
            
            if use_cfg:
                samples, _ = samples.chunk(2, dim=0)
            
            # 5. Decode
            samples = vae.decode(samples / 0.18215).sample.clamp(-1, 1)
            
            # 6. Visualize / Save
            img_vis = (img.float().detach().cpu() + 1) * 0.5
            edge_vis = edge.float().detach().cpu()
            edge_vis = (edge_vis + 1) * 0.5
            if edge_vis.shape[1] == 1:
                edge_vis = edge_vis.repeat(1, 3, 1, 1)
            gen_vis = (samples.float().detach().cpu() + 1) * 0.5
            
            for j in range(n):
                # 获取当前样本的类别标签和文件名
                class_id = y[j].item()
                if isinstance(filenames, list):
                    filename = filenames[j]
                else:
                    filename = filenames[j] if j < len(filenames) else f"sample_{cnt:05d}"
                
                # 为每个类别创建主文件夹和三个子文件夹
                class_dir = os.path.join(args.results_dir, f"class_{class_id}")
                origin_dir = os.path.join(class_dir, "origin")
                control_dir = os.path.join(class_dir, "control")
                gen_dir = os.path.join(class_dir, "gen")
                
                os.makedirs(origin_dir, exist_ok=True)
                os.makedirs(control_dir, exist_ok=True)
                os.makedirs(gen_dir, exist_ok=True)
                
                # 保存拼接后的图片（原有功能）
                grid = make_grid([img_vis[j], edge_vis[j], gen_vis[j]], nrow=3, padding=2)
                grid_path = os.path.join(class_dir, f"{filename}_rank{rank}.png")
                save_image(grid, grid_path)
                
                # 分别保存单独的图片
                save_image(img_vis[j], os.path.join(origin_dir, f"{filename}.png"))
                save_image(edge_vis[j], os.path.join(control_dir, f"{filename}.png"))
                save_image(gen_vis[j], os.path.join(gen_dir, f"{filename}.png"))
                
                cnt += 1
        
        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (i + 1) * args.batch_size * world_size / elapsed
            print(f"Processed batch {i+1}/{len(loader)}, {samples_per_sec:.2f} samples/sec")
    
    # 同步所有进程
    if use_distributed:
        dist.barrier()
    
    if rank == 0:
        total_time = time.time() - start_time
        total_samples = len(dataset)
        print(f"Done. Results saved to {args.results_dir}")
        print(f"Total time: {total_time:.2f}s, Average: {total_samples/total_time:.2f} samples/sec")
    
    if use_distributed:
        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_gen")
    parser.add_argument("--control-type", type=str, choices=["mask", "canny"], default="mask",
                        help="Control signal type: 'mask' for mask-based, 'canny' for canny-based, etc.")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use (1 for single GPU, >1 for multi-GPU)")
    parser.add_argument("--vae", type=str, choices=["mse", "ema"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=4.0)

    # Control-specific args
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained ControlSiT checkpoint (.pt)")
    parser.add_argument("--canny-low", type=int, default=80)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--no-sample-control", action="store_true")
        
    # Single-class convenience
    parser.add_argument("--fixed-class-id", type=int, default=-1)
    parser.add_argument("--single-folder", action="store_true")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
