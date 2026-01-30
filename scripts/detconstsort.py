# Author: Gosh et al., modified by F.P.J. de Kam (floris.de.kam@student.uva.nl)

from __future__ import annotations
import argparse
from pathlib import Path
import torch

from helpers.utils import get_dataset
from helpers.metrics import torch_load_any, compute_global_pi_from_graph
from baseline.detconstsort.detconstsort_helpers import detconstsort_rerank

def ensure_splits(dataset: str, seed: int, splits_dir: Path):
    """
    Makes splits similar to MORAL training pipeline, or loads them if they already exist.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / f"{dataset}_{seed}.pt"

    if splits_path.exists():
        _, splits = torch_load_any(splits_path)
        return splits, splits_path

                                        # get dataset and create splits
    _ = get_dataset(dataset, splits_dir, seed)
    if not splits_path.exists():
        raise RuntimeError(f"Splits file was not created: {splits_path.resolve()}")

    _, splits = torch_load_any(splits_path)
    return splits, splits_path


def load_outputs_and_groups(path: Path):
    """
    Outputs scores from model output file
    """
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, tuple) and len(obj) == 2:
        outputs, edge_groups = obj
        outputs = torch.as_tensor(outputs).view(-1).float()
        edge_groups = torch.as_tensor(edge_groups).view(-1).long()
        return outputs, edge_groups

    raise ValueError(f"Unexpected format in {path.name}")


def run_one(
    dataset: str,
    seed: int,
    run: int,
    model_string: str,
    outputs_dir: Path,
    splits_dir: Path,
    K: int,
):
    """
    One run of DETCONSTSORT reranking, inspired by MORAL training run. 
    Takes output from three GAE classifiers (also used in MORAL),
    reranks them, and saves final ranking.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

                                        # input file for this run
    input_path = outputs_dir / f"three_classifiers_{dataset}{model_string}_{run}.pt"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")
                                        #  load outputs and groups
    outputs, edge_groups = load_outputs_and_groups(input_path)

                                        # Dummy raw copy for consistency
    out_dummy = outputs_dir / f"three_classifiers_{dataset}_DETCONSTSORT_GAE_{seed}.pt"
    torch.save((outputs, edge_groups), out_dummy)

                                        # create/load splits
    splits, splits_path = ensure_splits(dataset, seed, splits_dir)

                                        # reconstruct labels: pos and neg
    test_pos = int(splits["test"]["edge"].size(0))
    test_neg = int(splits["test"]["edge_neg"].size(0))
    test_labels = torch.cat([torch.ones(test_pos), torch.zeros(test_neg)], dim=0).float()

    if test_labels.numel() != outputs.numel():
        raise ValueError(
            f"[run {run}] Length mismatch: test_labels={test_labels.numel()} vs outputs={outputs.numel()}.\n"
        )

                                        # compute pi from graph structure
    pi = compute_global_pi_from_graph(dataset)
                                        # ensure it is a 1D float tensor
    pi = torch.as_tensor(pi).view(-1).float()   

                                        # run DETCONSTSORT reranking, kmax = K-1
    final_scores, final_labels, final_groups, picked_idx = detconstsort_rerank(
        outputs=outputs,
        test_labels=test_labels,
        edge_sens_groups=edge_groups,
        pi=pi,
        K=K,
        kmax=K - 1,
    )

                                        # save final ranking
    out_final = outputs_dir / f"three_classifiers_{dataset}_DETCONSTSORT_GAE_{seed}_final_ranking.pt"
    torch.save(
        (final_scores, final_labels, picked_idx.long(), final_groups.long()),
        out_final,
    )
    print(f"[run {run}] saved final: {out_final.name}")


def main():
    """
    Lean version of MORAL training pipeline to run DETCONSTSORT reranking on
    outputs from three GAE classifiers.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="facebook")
    parser.add_argument("--seed_base", type=int, default=0, help="seed = seed_base + run (default seed=run)")
    project_root = Path(__file__).resolve().parents[1]

    args = parser.parse_args()
    args.runs = int(3)
    args.model_string = str("_MORAL_GAE")
    args.outputs_dir = project_root / "data" / "output"
    args.splits_dir = project_root / "data" / "splits"
    
    
    for run in range(args.runs):
        seed = args.seed_base + run
        run_one(
            dataset=args.dataset,
            seed=seed,
            run=run,
            model_string=args.model_string,
            outputs_dir=args.outputs_dir,
            splits_dir=args.splits_dir,
            K= 1000,
        )


if __name__ == "__main__":
    main()
