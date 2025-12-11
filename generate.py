import os
import argparse
import logging
import glob
import time
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from models import SiT_models
from controlnet_sit import ControlSiT
from diffusers.models import AutoencoderKL
from transport import create_transport, Sampler
from train_utils import parse_transport_args

# speed flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def center_crop_arr(pil_image, image_size, crop_ratio=1.0):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)

    if isinstance(crop_ratio, float):
        crop_ratio = crop_ratio
    else:
        crop_ratio = np.random.uniform(crop_ratio[0], crop_ratio[1])
    crop_size = int(round(image_size * crop_ratio))
    crop_y = (arr.shape[0] - crop_size) // 2
    crop_x = (arr.shape[1] - crop_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + crop_size, crop_x: crop_x + crop_size])

class ImageWithCanny(transforms.Compose):
    def __init__(self, image_size, low=100, high=200):
        self.image_size = image_size
        self.low = low
        self.high = high
        super().__init__([
            transforms.Lambda(lambda pil: center_crop_arr(pil, image_size, crop_ratio=1.0)),
            # transforms.RandomHorizontalFlip(), # Disable flip for deterministic generation if desired, or keep it
        ])
    
    def __call__(self, img):
        img_t = super().__call__(img)
        np_img = np.array(img)
        
        if np_img.ndim == 3:
            gray = (0.299 * np_img[..., 0] + 0.587 * np_img[..., 1] + 0.114 * np_img[..., 2]).astype(np.uint8)
        else:
            gray = np_img.astype(np.uint8)        
        try:
            import cv2
            edges = cv2.Canny(gray, self.low, self.high)
        except Exception:
            gy = np.zeros_like(gray, dtype=np.float32)
            gx = np.zeros_like(gray, dtype=np.float32)
            gy[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            gx[1:-1, :] = gray[2:, :] - gray[:-2, :]
            edges = (np.hypot(gx, gy) > 64).astype(np.uint8) * 255

        img_t = transforms.ToTensor()(img) # [0, 1]
        img_t = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_t) # [-1, 1]
        
        edge_t = torch.from_numpy(edges).float().unsqueeze(0) / 255.0 # [1, H, W]
        
        return img_t, edge_t

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.global_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    print(f"Starting generation with seed={args.global_seed}.")

    # Setup folders
    os.makedirs(args.results_dir, exist_ok=True)

    # Model backbone
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    base = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)

    # Initialize ControlNet
    control_model = ControlSiT(base, freeze_base=True).to(device)
    control_model.eval()

    # Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    if "ema" in checkpoint:
        print("Loading EMA weights from checkpoint...")
        control_model.load_state_dict(checkpoint["ema"])
    elif "model" in checkpoint:
        print("Loading model weights from checkpoint...")
        control_model.load_state_dict(checkpoint["model"])
    else:
        print("Loading weights from state dict...")
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
    print(f"Loading VAE: {args.vae}")
    vae = AutoencoderKL.from_pretrained(f'./sd-vae-ft-{args.vae}').to(device)
    vae.eval()
    
    # Data
    transform = ImageWithCanny(args.image_size, args.canny_low, args.canny_high)
    if args.single_folder:
        dataset = FlatFolderDataset(args.data_path, transform=transform)
    else:
        dataset = ImageFolder(args.data_path, transform=transform)
    
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False
    )
    
    print(f"Dataset contains {len(dataset)} images.")
    
    cnt = 0
    for i, ((img, edge), y) in enumerate(loader):
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
                # Save grid: Original | Edge | Generated
                grid = make_grid([img_vis[j], edge_vis[j], gen_vis[j]], nrow=3, padding=2)
                save_path = os.path.join(args.results_dir, f"sample_{cnt:05d}.png")
                save_image(grid, save_path)
                cnt += 1
        
        print(f"Processed batch {i+1}/{len(loader)}")

    print(f"Done. Results saved to {args.results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_gen")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
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
