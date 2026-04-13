from copy import deepcopy
from train import train
from metrics import compute_ntk, representation_similarity, sign_similarity, kernel_alignment
from tqdm import tqdm

def run_single_experiment(config, task, model_builder, inputs0):
    model = model_builder(config)
    _, activity0, _ = model(inputs0)
    K0 = compute_ntk(model, inputs0, config["task_mode"])
    logs = train(model, task, config)
    _, activity, _ = model(inputs0)
    Kf = compute_ntk(model, inputs0, config["task_mode"])
    results = {
        "weight_dist": logs[-1]["weight_dist"],
        "rep_sim": representation_similarity(activity0, activity),
        "sign_sim": sign_similarity(activity0, activity),
        "kernel_alignment": kernel_alignment(K0, Kf),
        "loss": logs[-1]["loss"],
    }
    print(results)
    return results

def run_with_lr_sweep(config, task, model_builder, inputs0):
    best_result = None
    best_loss = float("inf")
    for lr in config["lr_list"]:
        cfg = config.copy()
        cfg["lr"] = float(lr)
        result = run_single_experiment(cfg, task, model_builder, inputs0)
        if result["loss"] < best_loss:
            best_loss = result["loss"]
            best_result = result
    return best_result

def sweep_rank(config, task, model_builder, inputs0):
    results = {}
    for r in tqdm(config["rank_list"], desc="Rank Sweep", position=0):
        cfg = deepcopy(config)
        cfg["connectivity"]["type"] = "low_rank"
        cfg["connectivity"]["rank"] = r
        result = run_with_lr_sweep(cfg, task, model_builder, inputs0)
        results[f"rank_{r}"] = result
    return results

def sweep_spectral(config, task, model_builder, inputs0):
    results = {}
    for rho in tqdm(config["radius_list"], desc="Spectral Sweep", position=0):
        cfg = deepcopy(config)
        cfg["connectivity"]["type"] = "spectral"
        cfg["connectivity"]["spectral_radius"] = rho
        result = run_with_lr_sweep(cfg, task, model_builder, inputs0)
        results[f"radius_{rho}"] = result
    return results