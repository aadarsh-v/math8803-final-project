import torch

def weight_distance(W0, Wt):
    return torch.norm(W0 - Wt)

def spectral_radius(W):
    eigvals = torch.linalg.eigvals(W)
    return eigvals.abs().max().item()

def orthogonality_error(W):
    I = torch.eye(W.shape[0])
    return torch.norm(W.T @ W - I).item()

def representation_change(model, x, f0):
    with torch.no_grad():
        f_t = model(x)
    return torch.norm(f_t - f0).item()

def kernel_alignment(K0, Kf):
    return torch.sum(K0 * Kf) / (torch.norm(K0) * torch.norm(Kf))

def representation_similarity(H0, H):
    KR0 = H0[-1] @ H0[-1].T
    KR = H[-1] @ H[-1].T
    return torch.sum(KR * KR0) / (torch.norm(KR0) * torch.norm(KR))

def sign_similarity(H0, H):
    return (torch.sign(H0) == torch.sign(H)).float().mean()

def compute_ntk(model, inputs, task_mode):
    params = list(model.parameters()) 
    outputs, _, _ = model(inputs)
    grads = []
    
    if task_mode == "ngym":
        T, B, C = outputs.shape
        for t in range(T):
            for b in range(B):
                for k in range(C):
                    gs = torch.autograd.grad(
                        outputs[t, b, k], params, retain_graph=True
                    )
                    grads.append(torch.cat([g.flatten() for g in gs]))
    elif task_mode == "sMNIST":
        B, C = outputs[-1].shape
        for b in range(B):
            for k in range(C):
                gs = torch.autograd.grad(
                    outputs[-1, b, k], params, retain_graph=True
                )
                grads.append(torch.cat([g.flatten() for g in gs]))
    
    J = torch.stack(grads)
    return J @ J.T