"""Farthest point sampling with an optional fast path.

torch_cluster gives a fast CUDA kernel for `fps`, but it has no ROCm (AMD) build, and it is the only
thing PartCrafter needs from it. When torch_cluster is not installed we fall back to a plain PyTorch
version that returns the same kind of result (indices into the flattened points, `ratio` of each
batch, farthest point first). It is slower than the CUDA kernel but it runs on any device.
"""
import torch

try:
    from torch_cluster import fps as _fps_cuda
except ImportError:  # AMD/ROCm, Apple MPS, or CPU-only installs
    _fps_cuda = None


def _fps_torch(src, batch=None, ratio=0.5, random_start=True):
    if batch is None:
        batch = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
    picked = []
    for b in torch.unique(batch):
        idx = torch.nonzero(batch == b, as_tuple=False).flatten()
        pts = src[idx].float()
        n = pts.shape[0]
        k = max(1, int(round(n * ratio)))
        chosen = torch.empty(k, dtype=torch.long, device=src.device)
        dist = torch.full((n,), float("inf"), device=src.device)
        cur = int(torch.randint(n, (1,)).item()) if random_start else 0
        for i in range(k):
            chosen[i] = cur
            dist = torch.minimum(dist, ((pts - pts[cur]) ** 2).sum(-1))
            cur = int(torch.argmax(dist).item())
        picked.append(idx[chosen])
    return torch.cat(picked)


def fps(src, batch=None, ratio=0.5, random_start=True):
    if _fps_cuda is not None:
        return _fps_cuda(src, batch, ratio=ratio, random_start=random_start)
    return _fps_torch(src, batch, ratio=ratio, random_start=random_start)
