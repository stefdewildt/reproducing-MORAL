# Author: Mattos et al., modified by F.P.J. de Kam (floris.de.kam@student.uva.nl)

from __future__ import annotations
import csv
import math
from pathlib import Path
import re
from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING
import torch
from torch import Tensor
from torch_geometric.utils import coalesce
from torch_geometric.data import Data
if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

from .datasets import Facebook, German, Nba, Pokec_n, Pokec_z, Credit, AirTraffic, Chameleon, DPAH_dataset, CreditMultiClass
from .create_splits import create_moral_style_edge_splits


################################################################
# Helpers
################################################################
    
def _to_edge_index_from_adj(adj) -> torch.Tensor:
    """
    Convert various adjacency representations to edge_index [2, E].
    Supports torch sparse COO tensor, scipy sparse matrix, dense torch tensor / numpy array.
    """
                                        # torch sparse
    if torch.is_tensor(adj) and adj.is_sparse:
        adj = adj.coalesce()
        return adj.indices().long()

                                        # torch dense
    if torch.is_tensor(adj) and (not adj.is_sparse):
        row, col = (adj != 0).nonzero(as_tuple=True)
        return torch.stack([row, col], dim=0).long()

                                        # scipy sparse
    try:
        import scipy.sparse as sp
        if sp.issparse(adj):
            coo = adj.tocoo()
            return torch.tensor([coo.row, coo.col], dtype=torch.long)
    except Exception:
        pass

                                        # numpy dense
    try:
        import numpy as np
        if isinstance(adj, np.ndarray):
            row, col = (adj != 0).nonzero()
            return torch.tensor([row, col], dtype=torch.long)
    except Exception:
        pass

    raise TypeError(f"Unsupported adjacency type: {type(adj)}")


def _get_edge_index_and_num_nodes(dataset_obj, features) -> Tuple[torch.Tensor, int]:
    """
    Tries common patterns in custom dataset wrappers.
    
    Priority:
      1) dataset_obj.edge_index (already in PyG format)
      2) dataset_obj.adj or dataset_obj.adj() (sparse adjacency)
      3) dataset_obj.A or dataset_obj.A() (sometimes used)
    num_nodes is inferred from features.shape[0].
    """
                                        # infer num_nodes from features
    if torch.is_tensor(features):
        num_nodes = int(features.size(0))
    else:
        num_nodes = int(features.shape[0])

                                        # get edge_index directly
    if hasattr(dataset_obj, "edge_index"):
        edge_index = getattr(dataset_obj, "edge_index")
        if callable(edge_index):
            edge_index = edge_index()
        
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("dataset_obj.edge_index must have shape [2, E].")
        
        return edge_index, num_nodes

                                        # adjacency under common names
    for name in ("adj", "adjacency", "A"):
        if hasattr(dataset_obj, name):
            adj = getattr(dataset_obj, name)
            if callable(adj):
                adj = adj()
            
            edge_index = _to_edge_index_from_adj(adj)
            
            return edge_index, num_nodes

    raise AttributeError(
        "Could not find edge information. Expected dataset_obj.edge_index or dataset_obj.adj/adjacency/A."
    )


################################################################
# Our implementation of MORAL algorithm
################################################################

def moral_aggregate_algorithm(
    outputs: torch.Tensor,              
    test_labels: torch.Tensor,          
    edge_sens_groups: torch.Tensor,     
    pi: torch.Tensor,
    K: int = 1000,
    epsilon: float = 1e-12,
):
    """
    Implements Algorithm 1 from MATOS et al. paper: at each position t, pick the group j whose
    inclusion minimizes KL(q' || pi), then take the highest remaining scoring edge from that group.
    """
    device = outputs.device
    pi = pi.to(device=device, dtype=torch.float32)
                                        # normalise for numerical stability
    pi = pi / pi.sum().clamp_min(epsilon)

                                        # build candidate lists C_j: 
                                        # indices of items in each group, sorted by descending score
    candidates = []
    for j in range(int(pi.numel())):
                                        # get indices from sensitive group
        idx = (edge_sens_groups == j).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            candidates.append(idx)
            continue
                                        # sort on edge relevance score predicted by model
        sorted_idx = idx[torch.argsort(outputs[idx], descending=True)]
        candidates.append(sorted_idx)

                                        # pointers into each sorted list, works for multi-class
    ptr = [0 for _ in range(int(pi.numel()))]
                                        # exposure counts c
    counts = torch.zeros_like(pi)  

    final_scores = torch.empty((K,), device=device, dtype=outputs.dtype)
    final_labels = torch.empty((K,), device=device, dtype=test_labels.dtype)
    final_groups = torch.empty((K,), device=device, dtype=torch.long)
                                        # up to k
    for t in range(1, K + 1):
        best_kl = None
        best_group = None

        for j in range(len(candidates)):
                                        # check if C_j empty
            if ptr[j] >= candidates[j].numel():
                continue  

                                        # q' = (counts with +1 in group j) / t
            q_prime = counts.clone()
            q_prime[j] += 1.0
            q_prime = q_prime / float(t)

                                        # KL(q' || pi)
            kl = (q_prime * torch.log((q_prime + epsilon) / (pi + epsilon))).sum()
                                        # save best KL and corresponding group
            if best_kl is None or kl < best_kl:
                best_kl = kl
                best_group = j

        if best_group is None:
                                        # no candidates left at all
            final_scores = final_scores[: t - 1]
            final_labels = final_labels[: t - 1]
            final_groups = final_groups[: t - 1]
            break

                                        # take top remaining element from chosen group
        pick_idx = candidates[best_group][ptr[best_group]]
        ptr[best_group] += 1
                                        # update final output
        final_scores[t - 1] = outputs[pick_idx]
        final_labels[t - 1] = test_labels[pick_idx]
        final_groups[t - 1] = best_group

        counts[best_group] += 1.0

    return final_scores, final_labels, final_groups


################################################################
# Data processing helpers
################################################################

def gradients_per_epoch(
    train_edges: int, 
    batch_size: int, 
    sensitive_categories: int
) -> int:
    """
    Compute the number of gradient updates per epoch given the training edges,
    batch size per sensitive group, and number of sensitive categories.
    From formula in Matos et al. paper.
    """
                                        # sanity check, avoid division by zero
    if sensitive_categories < 2:
                                        # if there's 1 category, all edges are in the same group,
                                        # thus gradients per epoch is simply total edges / batch size
        return math.ceil(train_edges / batch_size)
                                        # grads per epoch
    grads_epoch = math.ceil(train_edges / (sensitive_categories* math.comb(sensitive_categories, 2) * batch_size))
    return grads_epoch


def print_graph_statistics(
    csv_out_dpah, 
    batch_size=1024, 
    sensitive_categories=2
) -> None:
    """
    Print graph statistics from CSV as a formatted table.
    """
    with Path(csv_out_dpah).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    if not rows:
        print("No data found in CSV file.")
        return
                                        # define columns with headers
    columns = [
        ("dataset", "Dataset"),
        ("nodes", "Nodes"),
        ("average_degree", "Avg Degree"),
        ("gradients_per_epoch", "Gradients/Epoch"),
        ("global_00", "Global π counts (00)"),
        ("global_01", "(01)"),
        ("global_11", "(11)"),
        ("global_p00", "Global π dist. (00)"),
        ("global_p01", "(01)"),
        ("global_p11", "(11)"),
        ("train_pos_00", "Train π counts (00)"),
        ("train_pos_01", "(01)"),
        ("train_pos_11", "(11)"),
        ("train_pos_p00", "Train π dist. (00)"),
        ("train_pos_p01", "(01)"),
        ("train_pos_p11", "(11)"),
        ("valid_pos_00", "Valid π counts (00)"),
        ("valid_pos_01", "(01)"),
        ("valid_pos_11", "(11)"),
        ("valid_pos_p00", "Valid π dist. (00)"),
        ("valid_pos_p01", "(01)"),
        ("valid_pos_p11", "(11)"),
        ("test_pos_00", "Test π counts (00)"),
        ("test_pos_01", "(01)"),
        ("test_pos_11", "(11)"),
        ("test_pos_p00", "Test π dist. (00)"),
        ("test_pos_p01", "(01)"),
        ("test_pos_p11", "(11)"),
        ("train_pos_neg_overlap", "Train pos/neg overlap"),
        ("valid_pos_neg_overlap", "Valid pos/neg overlap"),
        ("test_pos_neg_overlap", "Test pos/neg overlap"),
    ]
    
    col_keys = [c[0] for c in columns]
    col_headers = [c[1] for c in columns]
                                        # build table data
    table_data = []
    seen_bases = set()
    for row in rows:
                                        # deduplicate by dataset base name 
                                        # strip trailing _<seed> 
        ds = row.get('dataset', '')
                                        # Remove a trailing underscore + digits (seed) only
        base_ds = re.sub(r'_(\d+)$', '', ds)
                                        # skip duplicate datasets
        if base_ds in seen_bases:
            continue
        seen_bases.add(base_ds)

        row_data = {}
                                        # set extra fields
        for col_key, _ in columns:
            if col_key == "gradients_per_epoch":
                try:
                                        # calculate gradient per epoch
                    train_pos_e = int(row.get('train_pos_E', 0))
                    row_data[col_key] = gradients_per_epoch(train_pos_e, batch_size, sensitive_categories)
                except Exception:
                    row_data[col_key] = "N/A"
            elif col_key == "dataset":
                                        # set dataseet name
                row_data[col_key] = base_ds
            else:
                row_data[col_key] = row.get(col_key, "")
        
        table_data.append(row_data)
                                        # calculate column widths
    col_widths = {}
    for i, (col_key, col_header) in enumerate(columns):
        col_widths[i] = max(
            len(col_header),
            max(len(str(row_data.get(col_key, ""))) for row_data in table_data) if table_data else 0
        )
                                        # print header
    header = " | ".join(col_headers[i].ljust(col_widths[i]) for i in range(len(columns)))
    print(header)
    print("-" * len(header))
                                        # print rows
    for row_data in table_data:
        print(" | ".join(str(row_data.get(col_keys[i], "")).ljust(col_widths[i]) for i in range(len(columns))))

    
def _get_dataset_object(
    dataset, 
    G=None, 
    feature_dim_G=16, 
    seed_G=42
):
    """
    Get dataset object for corresponding string, or generate DPAH dataset with parameters
    """
    dataset_map: Dict[str, object] = {
        "facebook": Facebook,
        "german": German,
        "nba": Nba,
        "pokec_n": Pokec_n,
        "pokec_z": Pokec_z,
        "credit": Credit,
        "credit_multiclass": CreditMultiClass,
        "airtraffic": AirTraffic,
        "chameleon": Chameleon,
    }
                                        # generate DPAH dataset using Graph
    if dataset.startswith("dpah"):
        if G is None:
            raise ValueError("For dataset='dpah', you must pass a NetworkX graph via G=...")
        dataset_obj = DPAH_dataset(G=G, feature_dim=int(feature_dim_G), seed=int(seed_G))
    else:
                                        # get standard dataset
        try:
            dataset_obj = dataset_map[dataset]()
        except KeyError as exc:
            raise ValueError(f"Unknown dataset '{dataset}'.") from exc

    return dataset_obj


################################################################
# Original code with adaptations
################################################################

def to_torch_sparse_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tensor:
    """
    Convert an edge index representation into a sparse COO tensor.
    """
    if size is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        size = (num_nodes, num_nodes)
    elif isinstance(size, int):
        size = (size, size)

    num_rows, num_cols = size
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_rows, num_cols)
    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    sparse = torch.sparse_coo_tensor(edge_index, edge_attr, size=size, device=edge_index.device)
    return sparse.coalesce()


def get_dataset(
    dataset: str,
    splits_dir: Union[str, Path],
    seed: int,
    *,
    G: Optional["nx.DiGraph"] = None,
    feature_dim_G: int = 16,
    seed_G: int = 42,
    assume_undirected: bool = True,
) -> Tuple:
    """
    Load dataset tensors and generate splits.
    For the synthetic DPAH datasets, generates dataset using NetworkX graph
    """
                                        # check or generate path
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / f"{dataset}_{seed}.pt"
                                        # get dataset
    dataset_obj = _get_dataset_object(dataset, G=G, feature_dim_G=feature_dim_G, seed_G=seed_G)

                                        # get raw objects from wrapper
    features = dataset_obj.features()
    idx_train = dataset_obj.idx_train()
    idx_val = dataset_obj.idx_val()
    idx_test = dataset_obj.idx_test()
    labels = dataset_obj.labels()
    sens = dataset_obj.sens()
    sens_idx = dataset_obj.sens_idx()

                                        # ensure torch types / shapes
    if not torch.is_tensor(features):
        features = torch.tensor(features, dtype=torch.float)
    else:
        features = features.float()

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long)
    labels = labels.view(-1).long()

    if not torch.is_tensor(sens):
        sens = torch.tensor(sens)
    sens = sens.view(-1).long()
                                        # get edge index and number of nodes
    edge_index, num_nodes = _get_edge_index_and_num_nodes(dataset_obj, features)
    
                                        # boolean masks from index lists
    train_idx = torch.as_tensor(idx_train, dtype=torch.long)
    val_idx = torch.as_tensor(idx_val, dtype=torch.long)
    test_idx = torch.as_tensor(idx_test, dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    if train_idx.numel() > 0: train_mask[train_idx] = True
    if val_idx.numel() > 0: val_mask[val_idx] = True
    if test_idx.numel() > 0: test_mask[test_idx] = True
    
                                        # generate splits for this seed 
    splits = create_moral_style_edge_splits(
        edge_index=edge_index,
        sens=sens,
        num_nodes=num_nodes,
        seed=int(seed),
        ratios=(0.7, 0.1, 0.2),
        neg_per_pos=1,
        assume_undirected=assume_undirected,
    )

                                        # build a PyG Data object
    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    
    data.sens = sens
    data.sens_idx = torch.as_tensor(sens_idx)
                                        # save splits file
    torch.save((data, splits), splits_path)

                                        # build adjacency from training positive edges only
    adj = to_torch_sparse_tensor(splits["train"]["edge"].t())

    return adj, features, train_idx, val_idx, test_idx, labels, sens, sens_idx, data, splits