# Author: Rahman et al.

from __future__ import annotations
import random
from collections import defaultdict
from typing import List
from loguru import logger
import numpy as np
import torch
from gensim.models import Word2Vec

def _build_buckets(edge_index: torch.Tensor, sens: torch.Tensor, num_groups: int = 2):
    edge_index = edge_index.cpu()
    sens = sens.view(-1).cpu().long()
    n = int(sens.numel())

    buckets: List[List[List[int]]] = [[[] for _ in range(num_groups)] for _ in range(n)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for u, v in zip(src, dst):
        g = int(sens[v].item())
        if 0 <= g < num_groups:
            buckets[u][g].append(v)

    return buckets


def _fair_next(buckets, u: int) -> int | None:
    groups = [g for g, neigh in enumerate(buckets[u]) if len(neigh) > 0]
    if not groups:
        return None
    g = random.choice(groups)
    return random.choice(buckets[u][g])


def _generate_walks(
    edge_index: torch.Tensor,
    sens: torch.Tensor,
    num_walks: int,
    walk_len: int,
    seed: int,
) -> List[List[str]]:
    random.seed(seed)
    np.random.seed(seed)

    # binary only
    buckets = _build_buckets(edge_index, sens, num_groups=2)
    n = int(sens.view(-1).numel())
    nodes = list(range(n))
    walks: List[List[str]] = []

    for _ in range(num_walks):
        random.shuffle(nodes)
        for start in nodes:
            walk = [start]
            cur = start
            for _ in range(walk_len - 1):
                nxt = _fair_next(buckets, cur)
                if nxt is None:
                    break
                walk.append(nxt)
                cur = nxt
            walks.append([str(x) for x in walk])
    return walks


def train_fairwalk_embeddings(
    edge_index: torch.Tensor,
    sens: torch.Tensor,
    dim: int = 128,
    window: int = 10,
    num_walks: int = 20,
    walk_len: int = 80,
    epochs: int = 5,
    workers: int = 1, # amount of cpu threads -> set to 1 for reproducibility
    seed: int = 0,
) -> torch.Tensor:
    sens = sens.view(-1).long()
    n_unique = int(torch.unique(sens.cpu()).numel())
    if n_unique != 2:
        raise ValueError(f"FairWalk implementation expects binary sens (2 classes), got {n_unique} classes.")

    walks = _generate_walks(edge_index, sens, num_walks=num_walks, walk_len=walk_len, seed=seed)

    w2v = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=0,
        sg=1,
        hs=0,
        negative=5,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    emb = torch.empty((int(sens.numel()), dim), dtype=torch.float32)
    for i in range(int(sens.numel())):
        emb[i] = torch.from_numpy(w2v.wv[str(i)]).float()
        
    logger.info("FairWalk emb shape: {}", tuple(emb.shape))
    logger.info(
        "Emb stats: min={:.4f} max={:.4f} mean={:.4f} std={:.4f}",
        emb.min().item(), emb.max().item(), emb.mean().item(), emb.std().item()
    )


    return emb


@torch.no_grad()
def score_edges_dot(emb: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    edges = edges.long()
    u = edges[:, 0]
    v = edges[:, 1]
    return (emb[u] * emb[v]).sum(dim=1)
