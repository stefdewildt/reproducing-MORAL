# Author: F.P.J. de Kam (floris.de.kam@student.uva.nl)

from __future__ import annotations
import csv
import math
from pathlib import Path
import random
from typing import Dict, Optional, Tuple, Union, Sequence, TYPE_CHECKING
import torch
from torch import Tensor
import numpy as np
if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

from .create_splits import create_moral_style_edge_splits, canonicalize_undirected_edges, edge_group_ids
from .utils import _get_dataset_object, _get_edge_index_and_num_nodes


################################################################
# Metrics
################################################################

def compute_ndkl(
    ranking,
    pi,
    *,
    k: int | None = None,
    num_groups: int | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Compute NDKL for a ranked list. ranking can be either
    the dict returned by reconstruct_ranked_edges (expects key 'group'), 
    or a 1D tensor/list of group IDs per rank position.

    Definition:
      NDKL = (1/Z) * sum_{k=1..K} [ 1/log2(k+1) * D_KL(pi_hat_k || pi) ]
      Z    = sum_{k=1..K} 1/log2(k+1)
      pi is the target/global group distribution (length num_groups).
    """
                                        # get ranking
    if isinstance(ranking, dict):
        groups = ranking.get("group")
        if groups is None:
            raise ValueError("ranking dict must contain a 'group' key.")
    else:
        groups = ranking
                                        # make tensor of groups
    groups_t = torch.as_tensor(groups, dtype=torch.long).view(-1)
    if groups_t.numel() == 0:
        raise ValueError("ranking/groups is empty.")
                                        # set k, k = complete ranking if not defined
    if k is None:
        k_eff = int(groups_t.numel())
    else:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        k_eff = int(min(k, groups_t.numel()))
                                        # limit ranking up to k
    groups_t = groups_t[:k_eff]
                                        # determine number of groups in pi distribution
    pi_t = torch.as_tensor(pi, dtype=torch.float).view(-1)
    if num_groups is None:
        num_groups = int(pi_t.numel())
    else:
        num_groups = int(num_groups)
        
    if int(pi_t.numel()) != int(num_groups):
        raise ValueError(f"pi must have length {num_groups}, got {pi_t.numel()}.")
                                        # smooth pi so we never divide by 0 
                                        # in case a group is absent globally
    if eps and eps > 0.0:
        pi_t = pi_t.clamp_min(eps)
        pi_t = pi_t / pi_t.sum()
    else:
                                        # target distribution pi must be strictly positive 
                                        # wherever pi_hat can put mass, otherwise KL is infinite
        if bool((pi_t <= 0).any()):
            raise ValueError(
                "pi contains zeros/negatives; KL(pi_hat||pi) is undefined/infinite without smoothing. "
                "Pass eps>0 to smooth, or ensure pi>0 for all groups."
            )
        pi_t = pi_t / pi_t.sum()
                                        # empirical prefix distributions pi_hat_k 
                                        # for all k via cumulative one-hot counts
    one_hot = torch.zeros((k_eff, num_groups), dtype=torch.float)
    valid = (groups_t >= 0) & (groups_t < num_groups)
    if not bool(valid.all()):
        bad = groups_t[~valid].unique().tolist()
        raise ValueError(f"Found out-of-range group ids: {bad}. Expected 0..{num_groups-1}.")
    
    one_hot.scatter_(1, groups_t.view(-1, 1), 1.0)

    cum_counts = one_hot.cumsum(dim=0)
    denom = torch.arange(1, k_eff + 1, dtype=torch.float).view(-1, 1)
                                        # determine input pi
    pi_hat = cum_counts / denom
                                        # KL per prefix: sum_i p_i * log(p_i / q_i)
                                        # using natural log
    q = pi_t.view(1, -1).expand_as(pi_hat)
    p = pi_hat
    kl_terms = torch.where(p > 0, p * (p / q).log(), torch.zeros_like(p))
    kl_per_k = kl_terms.sum(dim=1)

                                        # discount weights: 1/log2(k+1) for k=1..K.
    weights = 1.0 / torch.log2(torch.arange(2, k_eff + 2, dtype=torch.float))
    z = float(weights.sum().item())
    if z == 0.0 or math.isinf(z) or math.isnan(z):
        raise ValueError("Invalid normalization constant Z.")
                                        # return final NDKL calculation
    return float((weights * kl_per_k).sum().item() / z)


def compute_awrf(
    ranking,
    pi,
    *,
    k: int | None = None,
    num_groups: int = 3,
    eps: float = 1e-12,
) -> float:
    """
    Attention-Weighted Rank Fairness (AWRF). ranking can be either:
    dict with key 'group', or 1D tensor/list of group IDs per rank position.

    AWRF = sum_g | E_g - pi_g |
    where
      E_g = sum_k alpha_k * 1[group_k == g]
      alpha_k ∝ 1 / log2(k+1), normalized
    """
                                        # get ranking
    if isinstance(ranking, dict):
        groups = ranking.get("group")
        if groups is None:
            raise ValueError("ranking dict must contain a 'group' key.")
    else:
        groups = ranking
                                        # convert to tensor
    groups_t = torch.as_tensor(groups, dtype=torch.long).view(-1)
    if groups_t.numel() == 0:
        raise ValueError("ranking/groups is empty.")
                                        # set k, k = complete ranking if not defined
    if k is None:
        k_eff = int(groups_t.numel())
    else:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        k_eff = int(min(k, groups_t.numel()))
                                        # limit ranking up to k
    groups_t = groups_t[:k_eff]
                                        # calculate target pi
    pi_t = torch.as_tensor(pi, dtype=torch.float).view(-1)
    if int(pi_t.numel()) != int(num_groups):
        raise ValueError(f"pi must have length {num_groups}, got {pi_t.numel()}.")
                                        # normalise pi
    pi_t = pi_t.clamp_min(eps)
    pi_t = pi_t / pi_t.sum()

                                        # attention weights alpha_k ∝ 1 / log2(k+1)
    weights = 1.0 / torch.log2(torch.arange(2, k_eff + 2, dtype=torch.float))
    weights = weights / weights.sum().clamp_min(eps)
                                        # calculate E_g
    exposure = torch.zeros(num_groups, dtype=torch.float)
    for i in range(k_eff):
        g = int(groups_t[i])
        if g < 0 or g >= num_groups:
            raise ValueError(f"Invalid group id {g}. Expected 0..{num_groups-1}.")
        exposure[g] += weights[i]
                                        # calculate awrf
    awrf = torch.sum(torch.abs(exposure - pi_t))
    return float(awrf.item())


def prec_at_k(ranking_file: Union[str, Path], k: int) -> float:
    """
    Compute precision@k from a saved ranking file. The order stored should represent the ranking to evaluate.
    """

    if k <= 0:
        raise ValueError("k must be a positive integer.")
                                        # load ranking file
    obj = torch.load(Path(ranking_file), map_location="cpu")

                                        # expect a (scores, labels, ...) tuple
    if isinstance(obj, tuple) and len(obj) >= 2:
        _, labels = obj[:2]
                                        # labels should be a 1D tensor of 0/1 indicating ground truth
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels = labels.view(-1).float()
                                        # calculate precision with applicable k
        k_eff = int(min(k, labels.numel()))
        if k_eff == 0:
            return 0.0
        precision = float(labels[:k_eff].sum().item() / k_eff)
        return precision

                                        # if only scores were saved, we cannot compute precision without labels
    if torch.is_tensor(obj):
        raise ValueError(
            "Use a '*_final_ranking.pt' file produced by the pipeline."
        )

    raise ValueError("Unsupported ranking file format")


def ndcg_at_k(
    ranking_file: Union[str, Path],
    k: int,
    *,
    gains: str = "exponential",  
    epsilon: float = 1e-12,
) -> float:
    """
    Compute NDCG@k for a saved ranking. ranking_file must contain at least (scores, relevance_labels, ...).
    relevance_labels may be graded (e.g. 0,1,2,3) OR binary (0/1).
    With binary labels, NDCG@k mostly reflects how early the positives appear
    """
    if k <= 0:
        raise ValueError("k must be positive.")
                                        # load ranking file
    obj = torch.load(Path(ranking_file), map_location="cpu")

    if not (isinstance(obj, tuple) and len(obj) >= 2):
        raise ValueError("Expected ranking file to contain at least (scores, labels).")
                                        # get scores and relevance labels
    scores, rel_labels = obj[:2]
                                        # convert to tensor if not tensor
    if not torch.is_tensor(rel_labels):
        rel_labels = torch.as_tensor(rel_labels)
                                        # determine applicable k
    rel_labels = rel_labels.view(-1).float()
    k_eff = int(min(k, rel_labels.numel()))

    if k_eff == 0:
        return 0.0
                                        # discount factors: 1 / log2(i+1)
    positions = torch.arange(1, k_eff + 1, dtype=torch.float)
    discounts = 1.0 / torch.log2(positions + 1.0)

                                        # gains exponential or linear
    rel_k = rel_labels[:k_eff]
    if gains == "exponential":
        gains_k = torch.pow(2.0, rel_k) - 1.0
    elif gains == "linear":
        gains_k = rel_k
    else:
        raise ValueError("gains must be 'linear' or 'exponential'.")
                                        # calculate discounted cumulative gain
    dcg = torch.sum(gains_k * discounts)

                                        # ideal dcg
    ideal_rel, _ = torch.sort(rel_labels, descending=True)
    ideal_rel_k = ideal_rel[:k_eff]
                                        # scale gain exponentially optionally
    if gains == "exponential":
        ideal_gains = torch.pow(2.0, ideal_rel_k) - 1.0
    else:
        ideal_gains = ideal_rel_k
                                        # calculate idcg
    idcg = torch.sum(ideal_gains * discounts)

    if idcg < epsilon:
        return 0.0
                                        # calculate ndcg
    return float((dcg / idcg).item())


def edge_homophily(edge_index: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate edge homophily (edges belong to same sensitive group)
    """
    u, v = edge_index
    return float(np.mean(s[u] == s[v]))


def random_baseline_homophily(s: np.ndarray) -> float:
    """
    Calculate randome homophily baseline
    """
    p = np.mean(s)  
    return float(p**2 + (1 - p)**2)


def excess_homophily(edge_index: np.ndarray, s: np.ndarray):
    """
    Determine excess homophily by substracting random baseline
    from edge homophily
    """
    h_edge = edge_homophily(edge_index, s)
    h_rand = random_baseline_homophily(s)
    delta_h = h_edge - h_rand
    return h_edge, h_rand, delta_h


def calculate_edge_homophily(adj, sens):
    """
    Print edge_homophily statistics
    """
                                        # make edge_index from adjecancy tensor
    if hasattr(adj, "_indices"):  
        edge_index = np.vstack([adj._indices()[0].numpy(), adj._indices()[1].numpy()])
    else:  
                                        # make edge_index from numpy
        row, col = np.nonzero(adj)
        edge_index = np.vstack([row, col])

    s = sens.numpy() if hasattr(sens, "numpy") else sens
                                        # calculate excess homophily
    h_edge, h_rand, delta_h = excess_homophily(edge_index, s)
                                        # print stats
    print(f"Edge homophily: {h_edge:.4f}")
    print(f"Random baseline: {h_rand:.4f}")
    print(f"Excess homophily Δh = {delta_h:.4f}")

    if delta_h < 0:
        print("→ Heterophilic relative to random mixing")
    elif delta_h > 0:
        print("→ Homophilic relative to random mixing")
    else:
        print("→ Approximately random mixing")


def dyadic_dp(
    ranking: Dict,
    top_k: Optional[int] = None,
    intra_groups: Optional[Sequence[int]] = None,
    inter_groups: Optional[Sequence[int]] = None,
) -> float:
    """
    Dyadic demographic parity gap on ranked edges. 
    Calculates | mean(prob | intra) - mean(prob | inter) |
    
    Dynamic behavior:
      - If intra_groups & inter_groups are provided: use them (works for any #edge-types).
      - Else: assume the common binary/3-edge-type encoding (0,2=intra ; 1=inter).
    """
                                        # get scores and sensitve group labels
    scores = torch.as_tensor(ranking["score"]).flatten().float()
    groups = torch.as_tensor(ranking["group"]).flatten().long()

    if scores.numel() != groups.numel():
        raise ValueError(f"score and group must match: {scores.numel()} vs {groups.numel()}")
                                        # limit to top k
    if top_k is not None:
        k = int(min(max(int(top_k), 0), scores.numel()))
        scores, groups = scores[:k], groups[:k]
                                        # normalise to [0,1] if not yet probabilities
    probs = torch.sigmoid(scores) if (scores.min() < 0 or scores.max() > 1) else scores

    if (intra_groups is None) != (inter_groups is None):
        raise ValueError("provide both intra_groups and inter_groups, or neither")

    if intra_groups is None:
                                        # default MORAL binary grouping with 3 edge types
        intra = (groups == 0) | (groups == 2)
        inter = (groups == 1)
    else:
                                        # get intra and inter groups
        intra_ids = torch.as_tensor(list(intra_groups), dtype=torch.long, device=groups.device)
        inter_ids = torch.as_tensor(list(inter_groups), dtype=torch.long, device=groups.device)
        intra = torch.isin(groups, intra_ids)
        inter = torch.isin(groups, inter_ids)

    if intra.sum() == 0 and inter.sum() == 0:
        raise ValueError(f"need >=1 item in each set: intra={int(intra.sum())}, inter={int(inter.sum())}")
                                        # return DP
    return float((probs[intra].mean() - probs[inter].mean()).abs().item())


################################################################
# Helpers for results processing
################################################################

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def torch_load_any(path: Path):
    """
    The project stores PyG Data objects in split files.pt, which requires
    weights_only=False. This is safe here because we only load local files
    in the user's workspace.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
                                        # older PyTorch versions don't have the weights_only argument.
        return torch.load(path, map_location="cpu")

def load_graph_from_splits_file(splits_pt: Path) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load edge_index/sens/num_nodes from an existing data/splits/<dataset>.pt.
    The file is expected to contain (data, splits) where data is a PyG Data object.
    """
                                        # load splits file and check format
    obj = torch_load_any(splits_pt)
    if not (isinstance(obj, tuple) and len(obj) == 2):
        raise ValueError(f"Unexpected format in splits file: {splits_pt}")

    data, _ = obj
    if not hasattr(data, "edge_index"):
        raise ValueError(f"Splits file does not contain a Data object with edge_index: {splits_pt}")

    edge_index = torch.as_tensor(data.edge_index, dtype=torch.long)
                                        # get sensitive attributes
    if hasattr(data, "sens"):
        sens = torch.as_tensor(data.sens, dtype=torch.long).view(-1)
    else:
        raise ValueError(
            f"Data object in splits file has no 'sens' attribute: {splits_pt}. "
            "Re-generate splits with utils.get_dataset() so data.sens is stored."
        )
                                        # get num nodes or calculate
    if getattr(data, "num_nodes", None) is not None:
        num_nodes = int(data.num_nodes)
    elif hasattr(data, "x") and data.x is not None:
        num_nodes = int(data.x.size(0))
    else:
        num_nodes = int(edge_index.max().item() + 1) if edge_index.numel() else sens.numel()

    return edge_index, sens, num_nodes


def reconstruct_ranked_edges(
    outputs: torch.Tensor,
    dataset_name: str,
    seed: int,
    k: int | None = None,
    neg_per_pos: int = 1,
    ratios=(0.7, 0.1, 0.2),
    splits_dir: str | Path = "../../data/splits",
    G: Optional["nx.DiGraph"] = None,
    feature_dim_G: int = 16,
    seed_G: int = 42,
    assume_undirected: bool = True,
):
    seed_everything(seed)
    
    splits_file = Path(splits_dir) / f"{dataset_name}_{seed}.pt"
    
    if splits_file.exists():
                                        # load pre-computed splits 
        obj = torch_load_any(splits_file)
        
        if not (isinstance(obj, tuple) and len(obj) == 2):
            raise ValueError(f"Unexpected format in splits file: {splits_file}")

        _, splits = obj
                                        # get graph statistics
        edge_index, sens, num_nodes = load_graph_from_splits_file(splits_file)
    else:
                                        # get correct dataset
        dataset_obj = _get_dataset_object(dataset_name, G=G, feature_dim_G=feature_dim_G, seed_G=seed_G)
        
        sens = dataset_obj.sens()
        features = dataset_obj.features()
        
        edge_index, num_nodes = _get_edge_index_and_num_nodes(dataset_obj, features)
                                        # recreate splits corresponding to the given seed
        splits = create_moral_style_edge_splits(
                edge_index=edge_index,
                sens=sens,
                num_nodes=num_nodes,
                seed=int(seed),
                ratios=ratios,
                neg_per_pos=int(neg_per_pos),
                assume_undirected=assume_undirected,
                from_recreate=True,
            )
                                        # get splits and corresponding labels
                                        # same procedure as training loop in moral
    test_split = splits["test"]
    test_edges = torch.cat([test_split["edge"], test_split["edge_neg"]], dim=0).long()
    test_labels = torch.cat(
        [torch.ones(test_split["edge"].size(0)), torch.zeros(test_split["edge_neg"].size(0))],
        dim=0,
    ).float()
                                        # get sensitive edge groups 
    num_classes = int(sens.max().item()) + 1
    
                                        # sort output based on predicted score
    scores_sorted, idx = outputs.sort(descending=True)

                                        # apply k cutoff if set
    k_eff = int(idx.numel()) if k is None else int(min(int(k), idx.numel()))
    idx = idx[:k_eff]
                                        # set final score and labels and nodes
    final_score = scores_sorted[:k_eff]
    final_label = test_labels[idx]
    final_u = test_edges[idx, 0]
    final_v = test_edges[idx, 1]
                                        # determine sensitive group of nodes
    sens_u = sens[final_u]
    sens_v = sens[final_v]
                                        # determine sensitive edge groups of edges
    final_edges = torch.stack([final_u, final_v], dim=1).long()
    final_groups = edge_group_ids(final_edges, sens, num_classes).long()

    return {
        "mode": "raw",
        "u": final_u,
        "v": final_v,
        "sens_u": sens_u,
        "sens_v": sens_v,
        "group": final_groups,
        "score": final_score,
        "label": final_label,
    }
    

def _mean_std(vals: list[float]):
    """
    Compute mean and stddev of a list of floats.
    """
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
    return m, s


def _mean_std_vec(vecs: list[Union[np.ndarray, torch.Tensor, Sequence[float]]]):
    """
    Compute mean and stddev of a list of vectors.
    """
    if not vecs:
        return None, None
    arr = np.stack(
        [np.asarray(v, dtype = float) for v in vecs],
         axis = 0)
    mean = arr.mean(axis = 0)
    std  = arr.std(axis = 0)
    return mean, std


def load_final_ranking_file(
    final_ranking_file: Union[str, Path],
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Load a *_final_ranking.pt produced by main.py. 
    Ranking file should contain a tuple with at least (final_output, final_labels).
    Optionally, it can also contain output_positions and final_groups.
    """
    obj = torch.load(Path(final_ranking_file), map_location="cpu")
    if not (isinstance(obj, tuple) and len(obj) >= 2):
        raise ValueError("Expected final_ranking_file to contain at least (final_output, final_labels, ...).")
                                        # read components
    final_output, final_labels = obj[:2]
    output_positions = obj[2] if len(obj) >= 3 else None
    final_groups = obj[3] if len(obj) >= 4 else None
                                        # convert to tensors if needed
    if not torch.is_tensor(final_output):
        final_output = torch.as_tensor(final_output)
    if not torch.is_tensor(final_labels):
        final_labels = torch.as_tensor(final_labels)
    if output_positions is not None and (not torch.is_tensor(output_positions)):
        output_positions = torch.as_tensor(output_positions)
    if final_groups is not None and (not torch.is_tensor(final_groups)):
        final_groups = torch.as_tensor(final_groups)

    return (
        final_output.view(-1).float(),
        final_labels.view(-1),
        output_positions.view(-1).long() if output_positions is not None else None,
        final_groups.view(-1).long() if final_groups is not None else None,
    )
   
    
def _load_outputs_file(path: Union[str, Path]) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Load outputs from a saved outputs file.
    """
    try:
        outputs = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
                                        # older PyTorch versions don't have the weights_only argument.
        outputs = torch.load(path, map_location="cpu")
                                        # convert to tensor if needed
    if not torch.is_tensor(outputs):
        outputs = torch.as_tensor(outputs[0])
    
    return outputs.view(-1).cpu()


def compute_global_pi_from_graph(dataset_name: str, assume_undirected=True) -> torch.Tensor:
    """
    Compute global pi from the graph structure of a dataset.
    Works for multi-class sensitive attributes.
    """
    dataset_obj = _get_dataset_object(dataset_name)
                                        # get sensitive attributes and convert
    sens = dataset_obj.sens()
    if not torch.is_tensor(sens):
        sens = torch.as_tensor(sens)
    sens = sens.view(-1).long()
                                        # get features and edge_index
    features = dataset_obj.features()
    edge_index, _ = _get_edge_index_and_num_nodes(dataset_obj, features)
                                        # canonicalize undirected edges if needed
    if assume_undirected:
        edge_index = canonicalize_undirected_edges(edge_index)
                                        # make edge groups
    num_classes = int(sens.max().item()) + 1
    edges = edge_index.t().contiguous()  
    groups = edge_group_ids(edges, sens, num_classes)
                                        # compute global pi
    num_groups = num_classes * (num_classes + 1) // 2
    counts = torch.bincount(groups, minlength=num_groups).float()
    pi = counts / counts.sum().clamp_min(1e-12)
    
    return pi


def _load_global_pi_map(graph_stats_csv: str | Path) -> dict[str, torch.Tensor]:
    """
    Load dataset -> pi from a graph_stats CSV. 
    Expects columns: dataset, global_p00, global_p01, global_p11
    """
    csv_path = Path(graph_stats_csv)
                                        # check if file exists
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
                                        
    pi_map: dict[str, torch.Tensor] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "global_p00", "global_p01", "global_p11"}
                                        # check if columns exist
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"graph_stats CSV must contain columns: {sorted(required)}")

        for row in reader:
                                        # get data from rows for datasets
            ds = (row.get("dataset") or "").strip()
            if not ds:
                continue
            p00 = float(row["global_p00"])
            p01 = float(row["global_p01"])
            p11 = float(row["global_p11"])
            pi = torch.tensor([p00, p01, p11], dtype=torch.float32)
                                        # normalize defensively
            pi = pi / pi.sum().clamp_min(1e-12)
            pi_map[ds] = pi

    if not pi_map:
        raise ValueError("No datasets found in graph_stats CSV")
    
    return pi_map


################################################################
# Main get results functions
################################################################

def get_results(
    datasets: Sequence[str],
    folder: str | Path,
    graph_stats_csv: str | Path,
    k: int,
    num_groups: int = 3,
    model_string: str = "_MORAL_GAE",
    splits_dir: str | Path = "data/splits",
    G: Optional["nx.DiGraph"] = None,
    feature_dim_G: int = 16,
    seed_G: int = 42,
    runs: int = 3,
    assume_undirected: bool = True,
    seed_reproduce: int = 0,
):
    """
    Compute metrics by scanning output ranking results folder. 
    Raw NDKL is computed from outputs files saved as (outputs, edge_sens_groups)
    Reranked NDKL is computed from final_ranking files using final_groups
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(str(folder))
    if k <= 0:
        raise ValueError("k must be > 0")
                                        # Load pi dict from graph
    pi_map = _load_global_pi_map(graph_stats_csv)

    results: dict[str, dict[str, tuple[Optional[float], Optional[float]]]] = {}
    accum: dict[str, dict[str, list[float]]] = {}
    intra_group_idx = None
    inter_group_idx = None

                                        # loop for raw statistics (before MORAL Greedy KL-algorithm)
    for dataset in datasets:
        for run in range(runs):
                                        # set variables
            seed = run + seed_reproduce
            dataset_search = dataset + f"_{seed_reproduce}"
            path = folder / f"three_classifiers_{dataset}{model_string}_{seed}.pt"
            
            if not path.exists():
                raise FileNotFoundError(f"Outputs file for dataset '{dataset}' not found at {path}.")
                                        # specific idx and pi not from csv for multiclass experiment
            if dataset == "credit_multiclass":
                pi = compute_global_pi_from_graph(dataset)
                intra_group_idx = [0,3,5]
                inter_group_idx = [1,2,4]
            else:
                if dataset_search not in pi_map:
                    raise ValueError(f"Dataset '{dataset_search}' not found in graph_stats CSV.")
                                        # set pi for dataset
                pi = pi_map[dataset_search]
                                        
            outputs = _load_outputs_file(path)
                                        # reconstruct the ranking using data and outputs
            rows = reconstruct_ranked_edges(
                outputs=outputs, 
                dataset_name=dataset, 
                seed=seed, 
                splits_dir=splits_dir, 
                G=G,
                feature_dim_G=feature_dim_G,
                seed_G=seed_G,
                assume_undirected=assume_undirected,
            )
                                        # raw ndkl
            accum.setdefault(dataset, {}).setdefault("raw_NDKL", []).append(
                compute_ndkl(
                    ranking = rows,
                    pi = pi,
                    k = int(k),
                    num_groups = num_groups
                )
            )
                                        # AWRF
            accum.setdefault(dataset, {}).setdefault("raw_AWRF", []).append(
                compute_awrf(
                    ranking = rows,
                    pi = pi,
                    k = int(k),
                    num_groups = num_groups
                )
            )
                                        # Demographic parity on full ranking
            accum.setdefault(dataset, {}).setdefault("raw_demographic_parity", []).append(
                dyadic_dp(
                    ranking = rows,
                    top_k = None,
                    intra_groups = intra_group_idx,
                    inter_groups = inter_group_idx
                )
            )
            
                                        # calculate num of groups for plotting
            groups = torch.as_tensor(rows["group"]).view(-1).long()
            k_eff = int(min(int(k), groups.numel()))
            num_groups_local = int(pi.numel()) 
                                        # apply top k
            topk = groups[:k_eff]
                                        # count edges in each group
            counts = torch.bincount(
                topk,
                minlength = num_groups_local
            ).cpu().numpy()
                                        # calculate pi hat and return pi hat from ranks for plotting
            pi_hat = counts / max(k_eff, 1)
            accum.setdefault(dataset, {}).setdefault("raw_group_counts_at_k", []).append(counts)
            accum.setdefault(dataset, {}).setdefault("raw_pi_hat_at_k", []).append(pi_hat)

                                        # reset seed
    seed = seed_reproduce
                                        # MORAL Reranked ranking edge list + prec/ndcg
    for dataset in datasets:
        for run in range(runs):
                                        # set variables
            seed = run + seed_reproduce
            path = folder / f"three_classifiers_{dataset}{model_string}_{seed}_final_ranking.pt"
                                        # specific pi and idx for multiclass dataset
            if dataset == "credit_multiclass":
                pi = compute_global_pi_from_graph(dataset)
                intra_group_idx = [0,3,5]
                inter_group_idx = [1,2,4]
            else:
                pi = pi_map[dataset + f"_{seed_reproduce}"]
                                        # load from final ranking file
            final_output, _, _, final_groups = load_final_ranking_file(path)
            if final_groups is None:
                raise ValueError(f"{path} does not contain final_groups")
            
                                        # for reranked NDKL/AWRF we only need the group sequence 
            ranking_groups = final_groups
                                        # add top-k compositions for plotting later
            groups = torch.as_tensor(ranking_groups).view(-1).long()
                                        # now also work for multi-class (by numel())
            k_eff = int(min(int(k), groups.numel()))
            num_groups_local = int(pi.numel())
                                        # calculate pi hat from ranking list with k applied
            topk = groups[:k_eff]
            counts = torch.bincount(topk, minlength=num_groups_local).cpu().numpy()
            pi_hat = counts / max(k_eff, 1)

            accum.setdefault(dataset, {}).setdefault("reranked_group_counts_at_k", []).append(counts)
            accum.setdefault(dataset, {}).setdefault("reranked_pi_hat_at_k", []).append(pi_hat)

                                        # for reranked DP we need scores and groups
            reranked_rows = {"score": final_output, "group": final_groups}
            
            try:
                                        # NDKL
                accum.setdefault(dataset, {}).setdefault("reranked_NDKL", []).append(
                    compute_ndkl(
                        ranking = ranking_groups,
                        pi = pi, k = int(k),
                        num_groups = int(pi.numel())
                    )
                )
                                        # AWRF
                accum.setdefault(dataset, {}).setdefault("reranked_AWRF", []).append(
                    compute_awrf(
                        ranking = ranking_groups,
                        pi = pi,
                        k = int(k),
                        num_groups = int(pi.numel())
                    )
                )
                                        # precision
                accum.setdefault(dataset, {}).setdefault("precision_at_k", []).append(prec_at_k(path, int(k)))
                                        # NDCG
                accum.setdefault(dataset, {}).setdefault("NDCG_at_k", []).append(ndcg_at_k(path, int(k)))
                                        # demographic parity on full ranking
                accum.setdefault(dataset, {}).setdefault("reranked_demographic_parity", []).append(
                    dyadic_dp(
                        ranking = reranked_rows,
                        top_k = None,
                        intra_groups = intra_group_idx,
                        inter_groups = inter_group_idx
                    )
                )
            except Exception:
                pass

    for dataset, metrics in accum.items():
        results[dataset] = {
            "raw_NDKL": _mean_std(metrics.get("raw_NDKL", [])),
            "reranked_NDKL": _mean_std(metrics.get("reranked_NDKL", [])),
            "raw_AWRF": _mean_std(metrics.get("raw_AWRF", [])),
            "reranked_AWRF": _mean_std(metrics.get("reranked_AWRF", [])),
            "raw_demographic_parity": _mean_std(metrics.get("raw_demographic_parity", [])),
            "reranked_demographic_parity": _mean_std(metrics.get("reranked_demographic_parity", [])),
            "precision_at_k": _mean_std(metrics.get("precision_at_k", [])),
            "NDCG_at_k": _mean_std(metrics.get("NDCG_at_k", [])),
        }
                                        # also include top-k compositions
        raw_counts_mean, raw_counts_std = _mean_std_vec(metrics.get("raw_group_counts_at_k", []))
        raw_pihat_mean,  raw_pihat_std  = _mean_std_vec(metrics.get("raw_pi_hat_at_k", []))
        rer_counts_mean, rer_counts_std = _mean_std_vec(metrics.get("reranked_group_counts_at_k", []))
        rer_pihat_mean,  rer_pihat_std  = _mean_std_vec(metrics.get("reranked_pi_hat_at_k", []))

        results[dataset].update({
            "raw_group_counts_at_k_mean": raw_counts_mean,
            "raw_group_counts_at_k_std":  raw_counts_std,
            "raw_pi_hat_at_k_mean":       raw_pihat_mean,
            "raw_pi_hat_at_k_std":        raw_pihat_std,
            "reranked_group_counts_at_k_mean": rer_counts_mean,
            "reranked_group_counts_at_k_std":  rer_counts_std,
            "reranked_pi_hat_at_k_mean":       rer_pihat_mean,
            "reranked_pi_hat_at_k_std":        rer_pihat_std,
        })

    return results


def fmt(ms):
    """
    Format for printing statistics
    """
    if ms is None or ms[0] is None or ms[1] is None:
        return "N/A"
    return f"{ms[0]:.4f} ± {ms[1]:.4f}"


def print_results(results, k):
    """
    Print results dictionairy
    """
                                        # column headers
    headers = [
        "Dataset",
        f"Raw NDKL@{k}",
        f"Reranked NDKL@{k}",
        f"Raw AWRF@{k}",
        f"Reranked AWRF@{k}",
        f"Raw DP",
        f"Reranked DP",
        f"Precision@{k}",
        f"NDCG@{k}",
    ]
                                        # prepare rows
    rows = []
    for dataset, metrics in results.items():
        row = [
            dataset,
            fmt(metrics.get("raw_NDKL")),
            fmt(metrics.get("reranked_NDKL")),
            fmt(metrics.get("raw_AWRF")),
            fmt(metrics.get("reranked_AWRF")),
            fmt(metrics.get("raw_demographic_parity")),
            fmt(metrics.get("reranked_demographic_parity")),
            fmt(metrics.get("precision_at_k")),
            fmt(metrics.get("NDCG_at_k")),
        ]
        rows.append(row)
                                        # compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

                                        # print header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
                                        # print rows
    for row in rows:
        line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(line)
        