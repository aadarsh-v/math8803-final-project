import torch

def build_connectivity(config, shape):
    t = config["type"]
    if t == "low_rank":
        return low_rank(shape, config["rank"])
    elif t == "spectral":
        return spectral_radius_init(shape, config["spectral_radius"])
    elif t == "orthogonal":
        return orthogonal_init(shape, config["alpha"])
    elif t == "random":
        return torch.randn(*shape)
    else:
        raise ValueError(f"Unknown connectivity type: {t}")

def low_rank(shape, rank):
    n, m = shape
    U = torch.randn(n, rank)
    V = torch.randn(rank, m)
    return (U @ V) / (rank ** 0.5)

def spectral_radius_init(shape, rho):
    W = torch.randn(*shape)
    eigvals = torch.linalg.eigvals(W)
    current = eigvals.abs().max()
    return (W * (rho / current)).real