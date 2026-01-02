from diffusers.models import AutoencoderKL
model_id = "stabilityai/sd-vae-ft-ema"
out_dir = "sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(model_id)
vae.save_pretrained(out_dir)
print(f"Saved VAE to {out_dir}")