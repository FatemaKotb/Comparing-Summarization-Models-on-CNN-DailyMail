import os
import torch

def save_ckpt(path, model, optimizer, step, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
        "cfg": cfg
    }, path)

def load_ckpt(path, model, optimizer=None):
    obj = torch.load(path, map_location="cpu")
    model.load_state_dict(obj["model"])
    if optimizer is not None and "optim" in obj:
        optimizer.load_state_dict(obj["optim"])
    return obj
