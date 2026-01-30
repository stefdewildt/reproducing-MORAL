# Author: F.P.J. de Kam (floris.de.kam@student.uva.nl)

import glob
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

from .create_splits import canonicalize_undirected_edges


################################################################
# Helpers: edge canonicalization
################################################################

def _ensure_e2(edges: torch.Tensor) -> torch.Tensor:
    """
    Ensure edges are shape [E, 2]. Accepts [2, E] or [E, 2].
    """
                                        # convert to tensor
    edges = torch.as_tensor(edges)
                                        # check shapes
    if edges.numel() == 0:
        return edges.reshape(0, 2).long()
    if edges.dim() != 2:
        raise ValueError(f"edges must be 2D, got {tuple(edges.shape)}")
    if edges.size(1) == 2:
        return edges.long()
    if edges.size(0) == 2:
        return edges.t().contiguous().long()

    raise ValueError(f"edges must have shape [E,2] or [2,E], got {tuple(edges.shape)}")


################################################################
# Edge stats determination and printing
################################################################

def edge_type_stats(
    edges: torch.Tensor,
    sens: torch.Tensor,
    *,
    undirected: bool = True,
) -> Dict[str, Any]:
    """
    Compute counts and normalized distribution over sensitive edge groups: 00, 01, 11
    Returns dict with counts and distribution.
    """
    sens = torch.as_tensor(sens).view(-1).long()
                                        # ensure shape and optionally make undirected
    edges_e2 = _ensure_e2(edges)
    if undirected:
                                        # canonicalize via shared helper (expects [2,E])
        edge_index = edges_e2.t().contiguous()
        edge_index = canonicalize_undirected_edges(edge_index)
        edges_e2 = edge_index.t().contiguous()

    if edges_e2.numel() == 0:
        return {
            "counts": {"00": 0, "01": 0, "11": 0, "E": 0},
            "dist": {"00": 0.0, "01": 0.0, "11": 0.0},
        }
                                        # get u, v and sensitive group
    u = edges_e2[:, 0]
    v = edges_e2[:, 1]
    su = sens[u]
    sv = sens[v]
                                        # determine edge counts
    t00 = int(((su == 0) & (sv == 0)).sum().item())
    t11 = int(((su == 1) & (sv == 1)).sum().item())
    t01 = int(((su != sv)).sum().item())
                                        # total edge counts
    E = int(edges_e2.size(0))
                                        # calculate distribution
    denom = float(max(E, 1))
    dist = {
        "00": float(t00 / denom),
        "01": float(t01 / denom),
        "11": float(t11 / denom),
    }

    return {
        "counts": {"00": t00, "01": t01, "11": t11, "E": E},
        "dist": dist,
    }


def _flatten_result(path: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the nested result dict into a single CSV-friendly row.
    """
                                        # write name
    dataset_name = Path(path).stem
    row: Dict[str, Any] = {"dataset": dataset_name, "path": str(path)}
                                        # get global pi
    global_pi = result.get("global_pi", {})
    g_counts = global_pi.get("counts", {})
    g_dist = global_pi.get("dist", {})
                                        # write general graph info
    row["nodes"] = global_pi.get("nodes")
    row["average_degree"] = global_pi.get("average_degree")
                                        # write distribution and counts
    row.update(
        {
            "global_E": g_counts.get("E"),
            "global_00": g_counts.get("00"),
            "global_01": g_counts.get("01"),
            "global_11": g_counts.get("11"),
            "global_p00": g_dist.get("00"),
            "global_p01": g_dist.get("01"),
            "global_p11": g_dist.get("11"),
        }
    )

    for split in ("train", "valid", "test"):
        s = result.get(split, {})
                                        # write shape info for specific split
        row[f"{split}_pos_shape"] = s.get("pos_shape")
        row[f"{split}_neg_shape"] = s.get("neg_shape")
        row[f"{split}_pos_neg_overlap"] = s.get("pos_neg_overlap")

        for kind in ("pos", "neg"):
            stats = s.get(kind, {})
                                        # get stats for split
            counts = stats.get("counts", {})
            dist = stats.get("dist", {})
            
            prefix = f"{split}_{kind}"
                                        # write for pos/neg counts and distribution
            row[f"{prefix}_E"] = counts.get("E")
            row[f"{prefix}_00"] = counts.get("00")
            row[f"{prefix}_01"] = counts.get("01")
            row[f"{prefix}_11"] = counts.get("11")
            row[f"{prefix}_p00"] = dist.get("00")
            row[f"{prefix}_p01"] = dist.get("01")
            row[f"{prefix}_p11"] = dist.get("11")

    return row


def _default_csv_fieldnames() -> List[str]:
    """
    Determine default csv fieldnames
    """
                                        # default fieldnames
    fieldnames = [
        "dataset",
        "path",
        "nodes",
        "average_degree",
        "global_E",
        "global_00",
        "global_01",
        "global_11",
        "global_p00",
        "global_p01",
        "global_p11",
    ]

    for split in ("train", "valid", "test"):
                                        # names for shape of each split
        fieldnames.extend(
            [
                f"{split}_pos_shape",
                f"{split}_neg_shape",
                f"{split}_pos_neg_overlap",
            ]
        )
        for kind in ("pos", "neg"):
            prefix = f"{split}_{kind}"
                                        # names for counts and distributions of split
            fieldnames.extend(
                [
                    f"{prefix}_E",
                    f"{prefix}_00",
                    f"{prefix}_01",
                    f"{prefix}_11",
                    f"{prefix}_p00",
                    f"{prefix}_p01",
                    f"{prefix}_p11",
                ]
            )
    return fieldnames


def _upsert_csv_row(
    csv_path: str, 
    row: Dict[str, Any], 
    *, 
    key: str = "dataset"
) -> None:
    """
    Upsert one row to a CSV file by key. Used to update graph statistics csv.

    - If a row with the same `key` value exists, it is replaced.
    - Otherwise, the row is appended.
    - The file is created (with header) if missing/empty.
    """
                                        # assure csv file exists or create
    csv_path = str(csv_path)
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
                                        # get fieldnames
    fieldnames = _default_csv_fieldnames()

                                        # row is expected to match fieldnames
    if (not p.exists()) or p.stat().st_size == 0:
                                        # write headers
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    rows: list[Dict[str, Any]] = []
    found = False

    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
                                        # if the existing file has a different schema, 
                                        # we still rewrite using our stable schema
        for existing in reader:
            if key in existing and existing.get(key) == str(row.get(key)):
                                        # update existing row with our key/values
                merged = dict(existing)
                merged.update({k: ("" if v is None else v) for k, v in row.items()})
                rows.append(merged)
                found = True
            else:
                rows.append(existing)
                                        # make new row if not found
    if not found:
        rows.append({k: ("" if v is None else v) for k, v in row.items()})
                                        # write all rows to file
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(
    path: str, 
    *, 
    csv_out: Optional[str] = None, 
    verbose = True
) -> Dict[str, Any]:
                                        # Load graph data and splits from splits file
    data, splits = torch.load(path, map_location="cpu", weights_only=False)
                                        # set sensitive attribute
    sens = getattr(data, "sens", None)
    if sens is None:
        raise ValueError("No data.sens found. Add it when saving, or load sens separately.")
    
    sens = torch.as_tensor(sens).view(-1).long()
                                        
    out: Dict[str, Any] = {}

                                        # π: global distribution from original graph edges
    if not hasattr(data, "edge_index"):
        raise ValueError("Loaded data has no edge_index.")
                                        # get stats for each edge type
    out["global_pi"] = edge_type_stats(data.edge_index, sens, undirected=True)
                                        # compute average degree and number of nodes
    out['global_pi']['average_degree'] = out["global_pi"]["counts"]["E"] * 2 / sens.size(0)
    out["global_pi"]['nodes'] = sens.size(0)

    if verbose:
        print("\nGLOBAL π (from data.edge_index, undirected+dedup)")
        print("  counts:", out["global_pi"]["counts"])
        print("  dist:  ", out["global_pi"]["dist"])
        print("  nodes:", out["global_pi"]['nodes'])
        print("  average degree:", out['global_pi']['average_degree'], "(E * 2 / N)")
                                        # per split: pos/neg distributions
    for split in ["train", "valid", "test"]:
        pos = splits[split]["edge"]
        neg = splits[split]["edge_neg"]

                                        # shape check, our pipeline expects [E, 2]
        pos_e2 = _ensure_e2(pos)
        neg_e2 = _ensure_e2(neg)
                                        # get stats for each split and pos/neg
        pos_stats = edge_type_stats(pos_e2, sens, undirected=True)
        neg_stats = edge_type_stats(neg_e2, sens, undirected=True)

                                         # overlap check (pos/neg should be disjoint)
        pos_set = set(map(tuple, pos_e2.tolist()))
        neg_set = set(map(tuple, neg_e2.tolist()))
        inter = len(pos_set & neg_set)
                                        # create dict
        out[split] = {
            "pos": pos_stats,
            "neg": neg_stats,
            "pos_shape": tuple(pos_e2.shape),
            "neg_shape": tuple(neg_e2.shape),
            "pos_neg_overlap": inter,
        }

        if verbose:
            print(f"\n{split}")
            print("  pos shape:", out[split]["pos_shape"], "neg shape:", out[split]["neg_shape"])
            print("  pos counts:", pos_stats["counts"], "pos dist:", pos_stats["dist"])
            print("  neg counts:", neg_stats["counts"], "neg dist:", neg_stats["dist"])
            print("  pos/neg overlap:", inter)
                                        # optionally output into csv 
    if csv_out is not None:
        row = _flatten_result(path, out)
        _upsert_csv_row(csv_out, row, key="dataset")

    return out


if __name__ == "__main__":
    for p in glob.glob("data/splits/*"):
        print(f"\n{'='*60}\nProcessing: {p}\n{'='*60}")
        _ = main(p, csv_out="data/splits/splits_summary.csv")
