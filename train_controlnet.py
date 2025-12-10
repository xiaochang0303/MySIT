"""
Train ControlNet-SiT
--------------------
Training script for ControlNetSiT, based on train.py.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import SiT_models
from controlnet_sit import ControlNetSiT
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
import wandb_utils

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad: # Only update trainable parameters (ControlNet parts)
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# Dummy dataset for demonstration (replace with your own dataset)
# This dataset should return (image, label, control_image)
class ControlImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        # TODO: Implement your own logic to get the control image (e.g. Canny edge)
        # For now, we just return the image itself as a placeholder for the control
        # In a real scenario, you would process 'img' to get 'control'
        control = img.copy() 
        return img, label, control

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name = f"{experiment_index:03d}-ControlNet-{model_string_name}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    # Load pre-trained SiT model
    sit_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    
    # Load SiT checkpoints
    if args.ckpt is None:
        # Try to download or find default
        ckpt_path = f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
        if not os.path.exists(ckpt_path):
             logger.info(f"Checkpoint {ckpt_path} not found. Please provide --ckpt argument.")
    else:
        ckpt_path = args.ckpt
        
    if os.path.exists(ckpt_path):
        state_dict = find_model(ckpt_path)
        if "model" in state_dict:
            sit_model.load_state_dict(state_dict["model"])
        else:
            sit_model.load_state_dict(state_dict)
        logger.info(f"Loaded SiT checkpoint from {ckpt_path}")
    
    # Initialize ControlNet wrapper
    model = ControlNetSiT(sit_model=sit_model, hint_channels=3)
    
    # EMA
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    model = model.to(device)
    # DDP wrapper
    # We only need to wrap the trainable parts or the whole thing?
    # Wrapping the whole thing is easier, but we need to make sure gradients are only computed for trainable params
    model = DDP(model, device_ids=[device], find_unused_parameters=True) # find_unused_parameters might be needed if some SiT parts are skipped

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)

    # Setup data:
    # Note: We need a transform that handles both image and control image
    # For simplicity here, we apply same transform to both in the dataset class or here
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = ControlImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    model.train()
    
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y, control in loader:
            x = x.to(device)
            y = y.to(device)
            control = control.to(device)
            
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # Control image usually stays in pixel space or is encoded by the ControlNet's own encoder
                # Our ControlNetSiT expects the raw control image (but normalized)
                
            model_kwargs = dict(y=y, control=control)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to SiT checkpoint")
    
    # Transport args
    parser.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    parser.add_argument("--loss-weight", type=str, default=None, choices=[None, "velocity", "likelihood"])
    parser.add_argument("--sample-eps", type=float)
    parser.add_argument("--train-eps", type=float)
    
    # ControlNet args
    parser.add_argument("--cfg-scale", type=float, default=1.0) # For sampling if implemented

    args = parser.parse_args()
    main(args)
