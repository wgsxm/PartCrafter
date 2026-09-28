"""Pure-PyTorch farthest-point sampling, drop-in for torch_cluster.fps.

MPS/CPU-compatible. Matches the torch_cluster signature used in
autoencoder_kl_triposg.py:  fps(pos, batch, ratio=..., random_start=...)
Returns a LongTensor of selected indices into the flattened `pos`.
"""
import torch


def fps(pos: torch.Tensor, batch: torch.Tensor, ratio: float = 0.25,
        random_start: bool = False) -> torch.Tensor:
    device = pos.device
    out = []
    # process each batch group independently (batch is sorted, contiguous groups)
    uniq = torch.unique(batch)
    for b in uniq.tolist():
        gmask = batch == b
        gidx = torch.nonzero(gmask, as_tuple=False).squeeze(1)  # global indices
        p = pos[gidx]                                           # (n,3)
        n = p.shape[0]
        k = max(1, int(round(n * ratio)))
        if k >= n:
            out.append(gidx)
            continue
        # greedy farthest-point selection
        sel = torch.empty(k, dtype=torch.long, device=device)
        if random_start:
            start = int(torch.randint(0, n, (1,), device=device).item())
        else:
            start = 0
        sel[0] = start
        dist = torch.full((n,), float('inf'), device=device)
        last = start
        for i in range(1, k):
            d = torch.sum((p - p[last]) ** 2, dim=1)
            dist = torch.minimum(dist, d)
            last = int(torch.argmax(dist).item())
            sel[i] = last
        out.append(gidx[sel])
    return torch.cat(out, dim=0)
