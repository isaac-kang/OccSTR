import os

import torch

from tools.utils.logging import get_logger


def _unwrap_model(m):
    # Peel DDP/DataParallel and torch.compile (OptimizedModule) wrappers so
    # state_dict keys stay raw (no "module." / "_orig_mod." prefix).
    if isinstance(m, (torch.nn.parallel.DistributedDataParallel,
                      torch.nn.DataParallel)):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    if isinstance(m, (torch.nn.parallel.DistributedDataParallel,
                      torch.nn.DataParallel)):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def save_ckpt(
    model,
    cfg,
    optimizer,
    lr_scheduler,
    epoch,
    global_step,
    metrics,
    is_best=False,
    logger=None,
    prefix=None,
):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
    """
    if logger is None:
        logger = get_logger()
    if prefix is None:
        if is_best:
            save_path = os.path.join(cfg["Global"]["output_dir"], "best.pth")
        else:
            save_path = os.path.join(cfg["Global"]["output_dir"], "latest.pth")
    else:
        save_path = os.path.join(cfg["Global"]["output_dir"], prefix + ".pth")
    state_dict = _unwrap_model(model).state_dict()
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": state_dict,
        "optimizer": None if is_best else optimizer.state_dict(),
        "scheduler": None if is_best else lr_scheduler.state_dict(),
        "config": cfg,
        "metrics": metrics,
    }
    torch.save(state, save_path)
    logger.info(f"save ckpt to {save_path}")


def load_ckpt(model, cfg, optimizer=None, lr_scheduler=None, logger=None, mode='train'):
    """
    Resume from saved checkpoints
    :param checkpoint_path: Checkpoint path to be resumed
    """
    if logger is None:
        logger = get_logger()
    checkpoints = cfg["Global"].get("checkpoints")
    pretrained_model = cfg["Global"].get("pretrained_model")

    status = {}
    if checkpoints and os.path.exists(checkpoints):
        checkpoint = torch.load(checkpoints, map_location=torch.device("cpu"), weights_only=False)
        _unwrap_model(model).load_state_dict(checkpoint["state_dict"], strict=True)
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        resume_scheduler = cfg["Global"].get("resume_scheduler", False)
        if lr_scheduler is not None and checkpoint.get("scheduler") is not None and resume_scheduler:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        elif lr_scheduler is not None:
            # Advance fresh scheduler to checkpoint's global_step
            # so LR follows the new schedule curve at the correct position
            resumed_step = checkpoint["global_step"]
            for _ in range(resumed_step):
                lr_scheduler.step()
            logger.info(f"checkpoint resume: lr_scheduler rebuilt with new schedule, "
                        f"advanced to step {resumed_step}, "
                        f"lr={lr_scheduler.get_last_lr()[0]:.6f}")
        logger.info(f"resume from checkpoint {checkpoints} (epoch {checkpoint['epoch']})")

        status["global_step"] = checkpoint["global_step"]
        status["epoch"] = checkpoint["epoch"] + 1
        status["metrics"] = checkpoint["metrics"]
    elif pretrained_model:
        pretrained_model = os.path.expanduser(pretrained_model)
        if not os.path.exists(pretrained_model):
            raise FileNotFoundError(f"pretrained_model not found: {pretrained_model}")
        fresh_params = cfg["Global"].get("fresh_params", None) if 'train' in mode else None
        load_pretrained_params(model, pretrained_model, logger, fresh_params=fresh_params)
        logger.info(f"finetune from checkpoint {pretrained_model}")
    else:
        logger.info("train from scratch")
    return status


def load_pretrained_params(model, pretrained_model, logger, fresh_params=None):
    if pretrained_model.endswith(".safetensors"):
        from safetensors.torch import load_file
        logger.info(f"Loading weights from safetensors: {pretrained_model}")
        checkpoint = load_file(pretrained_model)
    else:
        logger.info(f"Loading weights using torch.load: {pretrained_model}")
        checkpoint = torch.load(pretrained_model, map_location=torch.device("cpu"), weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if fresh_params:
        skipped = [k for k in state_dict if any(k.startswith(p) for p in fresh_params)]
        if skipped:
            logger.info(f"fresh_params: keeping random init for {skipped}")
        state_dict = {k: v for k, v in state_dict.items() if k not in skipped}

    target = _unwrap_model(model)
    target.load_state_dict(state_dict, strict=False)
    model_keys = target.state_dict().keys()
    for name in model_keys:
        if name not in state_dict:
            logger.info(f"{name} is not in pretrained model")

