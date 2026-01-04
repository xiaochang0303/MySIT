def none_or_str(value):
    if value == 'None':
        return None
    return value


def log_training_config(logger, args, title="Training Configuration"):
    """
    记录训练配置参数到日志

    Args:
        logger: logging.Logger 实例
        args: argparse.Namespace 参数对象
        title: 日志标题
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)

    # 按类别分组记录参数
    arg_groups = {
        "Data": ["data_path", "image_size", "num_classes", "val_ratio"],
        "Model": ["model", "ckpt", "vae"],
        "ControlNet": ["control_type", "use_lightweight", "light_rank", "light_shared_depth",
                       "noise_scale", "control_strength", "unfreeze_base"],
        "LoRA": ["lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules",
                 "lora_qkv", "lora_only", "use_lora"],
        "Training": ["epochs", "batch_size", "global_batch_size", "lr", "weight_decay",
                     "max_grad_norm", "warmup_steps", "ema_decay", "global_seed"],
        "Anti-Overfitting": ["strong_augment", "random_ctrl_strength", "use_early_stop",
                             "patience"],
        "Sampling": ["cfg_scale", "cfg_channels", "fixed_class_id", "sample_every"],
        "Transport": ["path_type", "prediction", "loss_weight", "train_eps", "sample_eps"],
        "Logging": ["log_every", "ckpt_every", "results_dir", "wandb"],
    }

    args_dict = vars(args)
    logged_keys = set()

    for group_name, keys in arg_groups.items():
        group_items = []
        for key in keys:
            if key in args_dict:
                value = args_dict[key]
                # 跳过 None 或默认值不重要的参数
                if value is not None:
                    group_items.append((key, value))
                    logged_keys.add(key)

        if group_items:
            logger.info(f"[{group_name}]")
            for key, value in group_items:
                logger.info(f"  {key}: {value}")

    # 记录未分组的参数
    other_items = [(k, v) for k, v in args_dict.items()
                   if k not in logged_keys and v is not None and not k.startswith('_')]
    if other_items:
        logger.info("[Other]")
        for key, value in other_items:
            logger.info(f"  {key}: {value}")

    logger.info("=" * 60)

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")