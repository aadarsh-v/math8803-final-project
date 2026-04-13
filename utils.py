import yaml
import torch
import random
import numpy as np
import os
import json

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For saving results

def tensor_to_python(obj):
    """Recursively convert tensors to python numbers/lists for JSON saving."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    else:
        return obj

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(results, save_dir="results", filename="results.json"):
    results = tensor_to_python(results)
    ensure_dir(save_dir)
    path = os.path.join(save_dir, filename)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

def save_tensor(tensor, path):
    ensure_dir(os.path.dirname(path))
    torch.save(tensor, path)

def load_tensor(path):
    return torch.load(path)

# Model helpers

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)