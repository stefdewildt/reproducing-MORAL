# Author: F.P.J. de Kam (floris.de.kam@student.uva.nl)

import math
import random
from typing import Dict, List, Tuple
import torch


################################################################
# Helpers: edge canonicalization
################################################################

def canonicalize_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary edge_index into a canonical undirected set: 
    remove self-loops (u == v), map (u, v) and (v, u) to the same representation by sorting endpoints, 
    remove duplicate edges. Output shape: [2, E]
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2

    u, v = edge_index[0], edge_index[1]

                                        # remove self-loops
    mask = u != v
    u, v = u[mask], v[mask]

                                        # canonical endpoint ordering to deduplicate undirected edges
    a = torch.minimum(u, v)
    b = torch.maximum(u, v)

    ei = torch.stack([a, b], dim=0)

                                        # deduplicate columns (unique edges)
    ei = torch.unique(ei, dim=1)
    return ei


def edge_hash(u: int, v: int, num_nodes: int) -> int:
    """
    Hash an edge (u, v) into a single integer key for O(1) set membership.
    Requires u < v (canonical form).
    """
    return u * num_nodes + v


################################################################
# Helpers: pair type, splitting, negative sampling
################################################################

def edge_group_ids(
    edges: torch.Tensor,
    sens: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Determine edge group ids based on sensitive attributes of nodes.

    Supports edges shaped as [E, 2] or [2, E]. Returns a 1D tensor of length E
    with group ids in [0, C*(C+1)//2 - 1], where C=num_classes.
    """
                                        # ensure tensor shapes
    edges_t = torch.as_tensor(edges).long()
    if edges_t.numel() == 0:
        return edges_t.new_empty((0,), dtype=torch.long)
    if edges_t.dim() != 2:
        raise ValueError(f"edges must be 2D, got {tuple(edges_t.shape)}")

                                        # accept [E,2] or [2,E]
    if edges_t.size(1) == 2:
        u = edges_t[:, 0]
        v = edges_t[:, 1]
    elif edges_t.size(0) == 2:
        u = edges_t[0]
        v = edges_t[1]
    else:
        raise ValueError(f"edges must have shape [E,2] or [2,E], got {tuple(edges_t.shape)}")

                                        # get sensitive attributes
    s = torch.as_tensor(sens).view(-1).long()
    a = s[u]
    b = s[v]
                                        # determine lower and higher class for each edge
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
                                        # triangular indexing for unordered pairs
    base = lo * int(num_classes) - (lo * (lo - 1)) // 2
    return (base + (hi - lo)).long()


def pair_type(u: int, v: int, sens: torch.Tensor) -> int:
    """
    Map a node pair (u,v) to a dyad group id based on binary sens attribute.
    """
                                        # keep binary signature; delegate to multi-class dyad groups
    sens_t = torch.as_tensor(sens)
    edges = torch.tensor([[u, v]], dtype=torch.long, device=sens_t.device)
    return int(edge_group_ids(edges, sens_t, num_classes=2).item())


################################################################
# Helpers: MORAL-style multiclass dyad groups
################################################################

def moral_num_groups(num_classes: int) -> int:
    """
    Return number of unordered dyad groups for num_classes sensitive classes.
    """
    return num_classes * (num_classes + 1) // 2


def moral_edge_group_id_uv(
    u: int, 
    v: int, 
    sens: torch.Tensor, 
    num_classes: int
) -> int:
    """
    Multi-class version for mapping edge (u,v) to dyad group id.
    Groups correspond to unordered class pairs (i <= j). Total groups: G = C*(C+1)//2.
    """
                                        # delegate to vectorized implementation
    sens_t = torch.as_tensor(sens)
    edges = torch.tensor([[u, v]], dtype=torch.long, device=sens_t.device)
    return int(edge_group_ids(edges, sens_t, num_classes=int(num_classes)).item())


def moral_decode_group_id(g_id: int, num_classes: int) -> Tuple[int, int]:
    """
    Decode dyad group id -> (lo, hi) where 0 <= lo <= hi < num_classes.
    """
    current = 0
    
    for lo in range(num_classes):
        width = num_classes - lo
                                        # check if g_id falls in this block
        if current <= g_id < current + width:
                                        # set hi accordingly
            hi = lo + (g_id - current)
            return lo, hi
                                        # move to next block
        current += width
    
    raise ValueError(f"Invalid group id {g_id} for num_classes={num_classes}")


def moral_build_nodes_by_class(sens: torch.Tensor, num_classes: int) -> List[List[int]]:
    """
    Return nodes grouped by sensitive class.
    """
    nodes_by_c: List[List[int]] = []
    for c in range(num_classes):
        nodes_by_c.append(torch.where(sens == c)[0].tolist())
    return nodes_by_c


def moral_sample_negative_for_group(
    g_id: int,
    *,
    nodes_by_class: List[List[int]],
    num_classes: int,
    num_nodes: int,
    existing: set,
    torch_gen: torch.Generator,
    assume_undirected: bool,
) -> Tuple[int, int]:
    """
    Sample one negative edge (u,v) whose dyad group == g_id. For undirected returns u < v.
    """
                                        # decode group id to class pair
    lo, hi = moral_decode_group_id(g_id, num_classes)
                                        # make lists of nodes in each class
    A = nodes_by_class[lo]
    B = nodes_by_class[hi]
    
    if len(A) == 0 or len(B) == 0:
        raise RuntimeError(f"No nodes available for class pair ({lo},{hi})")

    for i in range(10000):
                                        # sample u from class lo, v from class hi
        u = A[torch.randint(0, len(A), (1,), generator=torch_gen).item()]
        v = B[torch.randint(0, len(B), (1,), generator=torch_gen).item()]
        if u == v:
            continue
                                        # hash canonically to avoid collisions with reversed order
        a, b = (u, v) if u < v else (v, u)
                                        # get group hash
        h = edge_hash(a, b, num_nodes)
        if h in existing:
            continue
        return (a, b) if not assume_undirected else (a, b)

    raise RuntimeError(f"Failed to sample negative edge for group {g_id} after many tries.")


def moral_sample_negatives_for_split_multiclass(
    pos_edges: List[Tuple[int, int]],
    *,
    sens: torch.Tensor,
    num_classes: int,
    num_nodes: int,
    neg_per_pos: int,
    nodes_by_class: List[List[int]],
    existing: set,
    torch_gen: torch.Generator,
    assume_undirected: bool,
) -> List[Tuple[int, int]]:
    """
    Sample negatives matching the dyad distribution of pos_edges.
    """
    if len(pos_edges) == 0:
        return []

    num_groups = moral_num_groups(num_classes)
                                        # get positive group ids
    pos_groups = [moral_edge_group_id_uv(u, v, sens, num_classes) for (u, v) in pos_edges]
    counts = torch.bincount(torch.tensor(pos_groups), minlength=num_groups).float()
    total_neg = int(neg_per_pos) * int(len(pos_edges))
                                        # sample group ids for negatives according to positive distribution
    probs = counts / counts.sum().clamp_min(1e-12)
    neg_samples = torch.multinomial(probs, num_samples=total_neg, replacement=True, generator=torch_gen)
    neg_per_group = torch.bincount(neg_samples, minlength=num_groups).tolist()

    neg_edges: List[Tuple[int, int]] = []
                                        # for each group, sample required number of negatives
    for g_id, n_g in enumerate(neg_per_group):
        for _ in range(int(n_g)):
                                        # sample negative edge
            u, v = moral_sample_negative_for_group(
                g_id,
                nodes_by_class=nodes_by_class,
                num_classes=num_classes,
                num_nodes=num_nodes,
                existing=existing,
                torch_gen=torch_gen,
                assume_undirected=assume_undirected,
            )
                                        # mark as existing so we don't resample it
            existing.add(edge_hash(u, v, num_nodes))
            neg_edges.append((u, v))
    return neg_edges

def split_group(
    items: List[Tuple[int, int]], 
    ratios: Tuple[float, float, float], 
    generator: torch.Generator
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Randomly split a list into train/val/test according to ratios, deterministically under seed.
    """
    if len(items) == 0:
        return [], [], []
                                        # shuffle items deterministically
    perm = torch.randperm(len(items), generator=generator).tolist()
    items = [items[i] for i in perm]
                                        # compute split sizes
    n = len(items)
    n_train = int(math.floor(ratios[0] * n))
    n_val = int(math.floor(ratios[1] * n))
                                        # make splits
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def to_edge_tensor(edges: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Convert list of edge tuples to [E,2] tensor.
    """
    if len(edges) == 0:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)


@torch.no_grad()
def create_moral_style_edge_splits(
    edge_index: torch.Tensor,
    sens: torch.Tensor,
    num_nodes: int,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
    neg_per_pos: int = 1,
    assume_undirected: bool = True,
    from_recreate: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    MORAL-style splits with stratification by sensitive dyad type.

    Generalized to multiclass sensitive attribute:
      - If sens has C classes, dyad groups are unordered pairs (i<=j): G = C*(C+1)//2 groups.
      - Positives are stratified by dyad group id, then split within each group.
      - Negatives are sampled per split to match the positive dyad distribution.

    Outputs shape [E,2] edge tensors.
    """
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-9
    assert edge_index.size(0) == 2
    assert sens.numel() == num_nodes

                                        # Make deterministic generator for reproducibility
    torch_gen = torch.Generator()
    if not from_recreate:
        torch_gen.manual_seed(seed)
        random.seed(seed)
    
    sens = sens.view(-1).long().cpu()
    num_classes = int(sens.max().item()) + 1
    if num_classes < 2:
        raise ValueError("sens must contain at least 2 classes for stratification.")
                                        # determine number of dyad groups and nodes by class
    num_groups = moral_num_groups(num_classes)
    nodes_by_class = moral_build_nodes_by_class(sens, num_classes)

                                        # canonicalize positives
    if assume_undirected:
        pos_all = canonicalize_undirected_edges(edge_index).cpu()
    else:
                                        # remove self-loops and deduplicate
        u, v = edge_index[0].cpu(), edge_index[1].cpu()
        mask = u != v
        pos_all = torch.stack([u[mask], v[mask]], dim=0)
        pos_all = torch.unique(pos_all, dim=1)
                                        # existing positives set for collision-free negatives
    existing = set()
    u_list = pos_all[0].tolist()
    v_list = pos_all[1].tolist()

    if assume_undirected:
        for u, v in zip(u_list, v_list):
                                        # expects u < v
            existing.add(edge_hash(u, v, num_nodes))  
    else:
        for u, v in zip(u_list, v_list):
                                        # assure u < v for hashing
            a, b = (u, v) if u < v else (v, u)
            existing.add(edge_hash(a, b, num_nodes))
                                        
                                        # stratify positives by dyad group id
    grouped_pos: Dict[int, List[Tuple[int, int]]] = {g: [] for g in range(num_groups)}
    for u, v in zip(u_list, v_list):
                                        # assign positives to groups
        g = moral_edge_group_id_uv(u, v, sens, num_classes)
        grouped_pos[g].append((u, v))

    pos_split: Dict[str, List[Tuple[int, int]]] = {"train": [], "valid": [], "test": []}
    for g_id in range(num_groups):
                                        # split positives within group
        tr, va, te = split_group(grouped_pos[g_id], ratios=ratios, generator=torch_gen)
        pos_split["train"].extend(tr)
        pos_split["valid"].extend(va)
        pos_split["test"].extend(te)

                                        # shuffle within split (avoid group ordering)
    for split_name in ("train", "valid", "test"):
        if len(pos_split[split_name]) == 0:
            continue
                                        # random permutation
        permutation = torch.randperm(len(pos_split[split_name]), generator=torch_gen).tolist()
        pos_split[split_name] = [pos_split[split_name][i] for i in permutation]

                                        # negative sampling per split to match positive dyad distribution
    splits: Dict[str, Dict[str, torch.Tensor]] = {}
    for split_name in ("train", "valid", "test"):
        pos_edges = pos_split[split_name]
                                        # sample negatives
        neg_edges = moral_sample_negatives_for_split_multiclass(
            pos_edges,
            sens=sens,
            num_classes=num_classes,
            num_nodes=num_nodes,
            neg_per_pos=neg_per_pos,
            nodes_by_class=nodes_by_class,
            existing=existing,
            torch_gen=torch_gen,
            assume_undirected=assume_undirected,
        )

        splits[split_name] = {
            "edge": to_edge_tensor(pos_edges),
            "edge_neg": to_edge_tensor(neg_edges),
        }

    return splits
