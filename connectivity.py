import torch

def build_connectivity(config, shape):
    t = config["type"]
    if t == "low_rank":
        return low_rank(shape, config["rank"])
    elif t == "spectral":
        return spectral_radius_init(shape, config["spectral_radius"])
    elif t == "random":
        return torch.randn(*shape) / (shape[0] ** 0.5)
    else:
        raise ValueError(f"Unknown connectivity type: {t}")

def low_rank(shape, rank):
    n, m = shape
    W0 = torch.randn(n, m)
    U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
    W_rank = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    W_rank = W_rank / torch.norm(W_rank) * torch.norm(W0)
    return W_rank

def spectral_radius_init(shape, rho):
    n, _ = shape
    W = torch.randn(*shape) / (n ** 0.5)
    eigvals = torch.linalg.eigvals(W)
    current = eigvals.abs().max()
    W = W * (rho / current)
    return W.real