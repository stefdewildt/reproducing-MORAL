# Author: F.P.J. de Kam (floris.de.kam@student.rug.nl)

from __future__ import annotations
from pathlib import Path
import torch
from torch_geometric.data import Data
import argparse
from pathlib import Path
from loguru import logger
from codecarbon import EmissionsTracker

from baseline.fairwalk.fairwalk_helpers import train_fairwalk_embeddings, score_edges_dot
from helpers.utils import get_dataset, moral_aggregate_algorithm
from helpers.check_splits import main as check_splits
from helpers.create_splits import canonicalize_undirected_edges
from main import resolve_model_config, seed_everything, generate_array_greedy_dkl


def parse_args() -> argparse.Namespace:
    """
    Parse arguments for fairwalk, adaptation of main trainings loop 
    with default arguments from the Rahman et al. paper
    """
    parser = argparse.ArgumentParser(description="Train MORAL on a selected dataset.")
    parser.add_argument("--dataset", type=str, default="facebook", help="Dataset identifier.")
    parser.add_argument("--model", type=str, default="gae", choices={"gae", "ncn"}, help="Base encoder/decoder setup.")
    parser.add_argument(
    "--fair_model",
    type=str,
    default="fairwalk",
    choices={"moral", "fairwalk"},
    help="Does nothing but handy for logging purposes.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device string.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size per sensitive group.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimensionality of the encoders.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay used by Adam.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs with different seeds.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed used for reproducibility.")
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("../data/splits"),
        help="Directory containing pre-computed edge splits (train/valid/test).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["gcn", "gin", "sage"],
        default=None,
        help="Override encoder type (e.g. 'gcn', 'sage', 'gin').  "
            "If not set, uses encoder from --model config.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Count of X many layers for the encoder (default: 2).",
    )
    parser.add_argument(
        "--ranking_loss",
        action="store_true",
        help="Deprecated option kept for compatibility with older scripts.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Deprecated option kept for compatibility with older scripts.",
    )
    parser.add_argument(
        "--csv_out",
        type=Path,
        default=Path("../data/output/fairwalk_edge_distributions.csv"),
        help="CSV file to store graph + split statistics (upserted).",
    )
    parser.add_argument(
        "--walks",
        type=int,
        default=20,
        help="Number of random walks for FairWalk.",
    )
    parser.add_argument(
        "--walk_len",
        type=int,
        default=80,
        help="Length of each random walk for FairWalk.",
    )
    
    args = parser.parse_args()
    
    if args.layers < 1:
        parser.error("--layers must be >= 1")
    return args


def build_test_edges_labels(splits: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get test edges and labels from split dict
    """
    test_split = splits["test"]
                                        # Concatenate positive and negative edges
    test_edges = torch.cat([test_split["edge"], test_split["edge_neg"]], dim=0)
    test_labels = torch.cat(
        [torch.ones(test_split["edge"].size(0)), torch.zeros(test_split["edge_neg"].size(0))],
        dim=0,
    )
    return test_edges, test_labels

def edge_groups_from_sens(test_edges: torch.Tensor, sens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Make edge groups from sensitive attribute value
    """
    sens_vec = sens.view(-1).long()
                                        # get nodes
    u = test_edges[:, 0].long()
    v = test_edges[:, 1].long()
                                        # make sensitive group
    edge_sens_groups = (sens_vec[u] + sens_vec[v]).long()
    sens_bin = sens_vec[u]  
    return edge_sens_groups, sens_bin

def fairwalk_scores(
    adj, 
    sens: torch.Tensor, 
    test_edges: torch.Tensor, 
    *, 
    dim: int, 
    seed: int, 
    device: str, 
    args
) -> torch.Tensor:
    """
    Determine fair walk scores bij training embeddings and calculating
    edge score with dot product.
    """
                                        # get train edges
    edge_index_train = adj.coalesce().indices() 
                                        # train embeddings
    emb = train_fairwalk_embeddings(
        edge_index=edge_index_train,
        sens=sens.view(-1).long(),
        dim=dim,
        num_walks=args.walks,
        walk_len=args.walk_len,
        epochs=args.epochs,
        workers=1,
        seed=seed,
    ).to(device)
    
    return score_edges_dot(emb, test_edges.to(device)).cpu()


def run_single(args: argparse.Namespace, run: int) -> None:
    """
    Single training run, adapted from MORAL training loop for fairwalk.
    """
                                        # set seeds
    seed = args.seed + run
    seed_everything(seed)
    logger.info("Run {}/{} — seed={}", run + 1, args.runs, seed)
    K = 1000
                                        # track emissions
    tracker = EmissionsTracker(
        project_name="FAIRWALK",
        experiment_id=f"{args.dataset}_{args.model}_run{run}",
        output_dir="../emissions",
        log_level="error",
    )
    tracker.start()
                                        # get dataset and generate splits
    adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits = get_dataset(
        args.dataset,
        args.splits_dir,
        seed,
    )
    
                                        # save graph statistics to csv
    _edge_distributions = check_splits(
        str(args.splits_dir / (args.dataset + f"_{seed}.pt")), 
        csv_out=str(args.csv_out), 
        verbose=False
    )

    labels = labels.cpu()
    sens = sens.cpu()

                                        # encoder type
    model_cfg = resolve_model_config(args.model)
    if args.encoder is not None:
        model_cfg["encoder"] = args.encoder
                                        # get test edges, labels and sensitive groups
    test_edges, test_labels = build_test_edges_labels(splits)
    edge_sens_groups, sens_bin = edge_groups_from_sens(test_edges, sens)

                                        # FairWalk is only for binary sensitive attributes
    n_unique = int(torch.unique(sens.view(-1)).numel())
    if n_unique > 2:
        raise ValueError(
            f"FairWalk is enabled but sensitive attribute has {n_unique} classes. "
            f"FairWalk baseline is intended for binary only."
        )

                                        # produce outputs by training fairwalk model
    outputs = fairwalk_scores(
        adj, sens, test_edges,
        dim=args.hidden_dim,
        seed=seed,
        device=args.device,
        args=args,
    ).cpu()

    run_suffix = f"{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}_{run}"
    out_dir = Path("../data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
                                        # save raw outputs
    torch.save(
        (outputs, edge_sens_groups.long()),
        out_dir / f"three_classifiers_{run_suffix}.pt",
    )
                                        # Need data.edge_index for global pi / output_positions 
                                        # if we rerank with MORAL
    if not isinstance(data, Data) or data.edge_index is None:
        logger.warning("Unexpected data type or missing edge_index; skipping ranking export.")
        return
                                        # canonicalize edges
    edge_index_canon = canonicalize_undirected_edges(data.edge_index)
                                        # convert to [E,2] so it matches edge_groups_from_sens signature
    global_edges = edge_index_canon.t().contiguous()
    global_groups, _ = edge_groups_from_sens(global_edges, sens)
                                        # compute global pi 
    num_groups = int(global_groups.max().item()) + 1
    pi_t = torch.bincount(global_groups.long(), minlength=num_groups).float()
    pi = (pi_t / pi_t.sum().clamp_min(1e-12)).cpu()

                                        # final ranking output, saved for compatibility
    output_positions = torch.as_tensor(
        generate_array_greedy_dkl(K, pi.numpy()),
        dtype=torch.long,
    )

                                        # MORAL reranking / aggregation
    final_output, final_labels, final_groups = moral_aggregate_algorithm(
        outputs=outputs,
        test_labels=test_labels,
        edge_sens_groups=edge_sens_groups,
        pi=pi.to(dtype=torch.float32),
        K=K,
    )
                                        # save MORAL output
    torch.save(
        (final_output, final_labels, output_positions.long(), final_groups.long()),
        out_dir / f"three_classifiers_{run_suffix}_final_ranking.pt",
    )

    emissions = tracker.stop()
    logger.info("Run {} emissions: {:.6f} kg CO₂", run + 1, emissions)

    logger.success("Finished run {}/{}", run + 1, args.runs)


def main() -> None:
    args = parse_args()
    logger.info("Processing dataset '{}' with model '{}'.", args.dataset, args.model)
    for run in range(args.runs):
        run_single(args, run)
        

if __name__ == "__main__":
    main()