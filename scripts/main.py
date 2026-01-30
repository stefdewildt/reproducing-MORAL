# Author: Mattos et al., modified by F.P.J. de Kam (floris.de.kam@student.uva.nl)

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data
from codecarbon import EmissionsTracker

from models.moral import MORAL
from models.moral_multiclass import MORALMultiClass
from helpers.utils import get_dataset, moral_aggregate_algorithm
from helpers.check_splits import main as check_splits
from helpers.create_splits import canonicalize_undirected_edges
from helpers.metrics import edge_group_ids

try:
    import networkx as nx
except Exception:  
    nx = None
    
################################################################
# Original helper functions
################################################################

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_array_greedy_dkl(n: int, distribution: np.ndarray) -> torch.Tensor:
    """
    Generate indices that approximate a target distribution for every prefix.
    """

    actual_counts = np.zeros_like(distribution)
    result = []

    for i in range(n):
        if i == 0:
            choice = int(np.argmax(distribution))
        else:
            desired = distribution * i
            deficit = desired - actual_counts
            choice = int(np.argmax(deficit))
        result.append(choice)
        actual_counts[choice] += 1

    return torch.tensor(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MORAL on a selected dataset.")
    parser.add_argument("--dataset", type=str, default="facebook", help="Dataset identifier.")
    parser.add_argument("--model", type=str, default="gae", choices={"gae", "ncn"}, help="Base encoder/decoder setup.")
    parser.add_argument("--fair_model", type=str, default="moral", help="Name of the fairness method (for logging only).")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device string.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size per sensitive group.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
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
        "--baseline",
        action="store_true",
        help="Deprecated option kept for compatibility with older scripts.",
    )
                                        # added arguments
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
        "--csv_out",
        type=Path,
        default=Path("../data/output/MORAL_edge_distributions.csv"),
        help="CSV file to store graph + split statistics (upserted).",
    )
    args = parser.parse_args()
    if args.layers < 1:
        parser.error("--layers must be >= 1")
    return args


def resolve_model_config(model_name: str) -> Dict[str, str]:
    mapping = {
        "gae": {"encoder": "gcn", "decoder": "gae"},
        "ncn": {"encoder": "gcn", "decoder": "mlp"},
    }
    try:
        return mapping[model_name]
    except KeyError as exc:  # pragma: no cover - safeguarded by argparse choices
        raise ValueError(f"Unsupported model '{model_name}'.") from exc


################################################################
# Main training loop with our modifications
################################################################

def run_single(
    args: argparse.Namespace,
    run: int,
    *,
    G: Optional["nx.DiGraph"] = None, 
    feature_dim_G: int = 16,
    seed_G: int = 42,
    assume_undirected: bool = True,
) -> None:
                                        # set seed for this run
    seed = args.seed + run
    seed_everything(seed)
    logger.info("Run {}/{} — seed={}", run + 1, args.runs, seed)
                                        # check if multiclass dataset
    multiclass = args.dataset.lower() == "credit_multiclass"
                                        # start emissions tracker
    tracker = EmissionsTracker(
        project_name="MORAL",
        experiment_id=f"{args.dataset}_{args.model}_run{run}",
        output_dir="../emissions",
        log_level="error",
    )
    tracker.start()

                                        # load dataset and generate splits
    adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits = get_dataset(
        args.dataset,
        args.splits_dir,
        seed,
        G=G,
        feature_dim_G=feature_dim_G,
        seed_G=seed_G,
        assume_undirected=assume_undirected,
    )

                                        # save graph statistics to csv
    if not multiclass:
        _edge_distributions = check_splits(
            str(args.splits_dir / (args.dataset + f"_{seed}.pt")),
            csv_out=str(args.csv_out), 
            verbose=False
        )
    else:
        logger.warning("check_splits skipped for credit_multiclass (binary-only stats).")


    labels = labels.cpu()
    sens = sens.cpu()

                                        # set encoder type
    model_cfg = resolve_model_config(args.model)
    if args.encoder is not None:
        model_cfg["encoder"] = args.encoder
                                        # default for layers is 2 if not set
    num_layers = args.layers
                                        # initialize original MORAL or MORALMultiClass
    if multiclass:
        model = MORALMultiClass(
            adj=adj,
            features=features,
            labels=labels,
            idx_train=idx_train.long(),
            idx_val=idx_val.long(),
            idx_test=idx_test.long(),
            sens=sens,                 
            sens_idx=sens_idx,
            edge_splits=splits,
            dataset_name=args.dataset,
            num_hidden=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            encoder=model_cfg["encoder"],
            encoder_layers=num_layers,
            decoder=model_cfg["decoder"],
            batch_size=args.batch_size,
            device=args.device,
            num_sens_classes=3,        
        )
    else:
                                        # initialize original MORAL
        model = MORAL(
            adj=adj,
            features=features,
            labels=labels,
            idx_train=idx_train.long(),
            idx_val=idx_val.long(),
            idx_test=idx_test.long(),
            sens=sens,
            sens_idx=sens_idx,
            edge_splits=splits,
            dataset_name=args.dataset,
            num_hidden=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            encoder=model_cfg["encoder"],
            decoder=model_cfg["decoder"],
            encoder_layers=num_layers,
            batch_size=args.batch_size,
            device=args.device,
        )
                                        # log model info
    logger.info("[CHECK] model type: {}", type(model).__name__)
    logger.info("[CHECK] sens unique: {}", torch.unique(sens).tolist())
    if hasattr(model, "num_groups"):
        logger.info("[CHECK] num_groups: {}", model.num_groups)
    
    logger.info("Training model…")

                                        # train model
    model.fit(epochs=args.epochs)
                                        # make predictions
    logger.info("Running inference on the test split…")
    outputs = model.predict().cpu()

    run_suffix = f"{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}_{run}"

    if not isinstance(data, Data):
        logger.warning("Unexpected data type returned by get_dataset; skipping ranking export.")
        return
    
    if data.edge_index is None:
        logger.warning("edge_index is None; skipping ranking export.")
        return
    
                                        # changed: use canonicalized undirected edges 
                                        # for global pi calculation
    edge_index_canon = canonicalize_undirected_edges(data.edge_index)
    sens_vec = sens.view(-1).long()  

    u, v = edge_index_canon[0], edge_index_canon[1]

    if multiclass:
        num_classes = int(sens_vec.max().item()) + 1  # 3
        num_groups = num_classes * (num_classes + 1) // 2  # 6
        canon_edges = edge_index_canon.t().contiguous()     # [E,2]
        edge_groups = edge_group_ids(canon_edges, sens_vec, num_classes)  # 0..5
        pi_t = torch.bincount(edge_groups, minlength=num_groups).float()
        pi = (pi_t / pi_t.sum()).cpu().numpy()
    else:
                                        # changed: pi calculation on undirected edges
                                        # and senstive edge groups
        edge_groups = sens_vec[u] + sens_vec[v]
        pi_t = torch.bincount(edge_groups, minlength=3).float()
        pi = (pi_t / pi_t.sum()).cpu().numpy()

    K = 1000
    
                                        # keep calculation of output_positions for compability
    output_positions = generate_array_greedy_dkl(K, pi)
                                        # get test edges and labels
    test_split = splits["test"]
    test_edges = torch.cat([test_split["edge"], test_split["edge_neg"]], dim=0)
    test_labels = torch.cat(
        [torch.ones(test_split["edge"].size(0)), torch.zeros(test_split["edge_neg"].size(0))],
        dim=0,
    )
    
    if multiclass:
                                        # get sensitive test edge groups for multiclass
        num_classes = sens.max().item() + 1  
        edge_sens_groups = edge_group_ids(test_edges.long(), sens_vec, num_classes)  
    else:
                                        # get sensitive test edge groups for binary
        edge_sens_groups = sens[test_edges].sum(dim=1)    

    torch.save((outputs, edge_sens_groups.long()), f"../data/output/three_classifiers_{run_suffix}.pt")

                                        # changed: use moral_aggregate_algorithm instead of original greedy_dkl_rerank
                                        # to get final ranking based on KL minimization
    final_output, final_labels, final_groups = moral_aggregate_algorithm(
        outputs=outputs,
        test_labels=test_labels,
        edge_sens_groups=edge_sens_groups,
        pi=torch.tensor(pi, dtype=torch.float32),
        K=K,
    )
    
                                        # added: save output_positions and final_groups to calculate reranked NDKL
    torch.save((final_output, final_labels, output_positions.long(), final_groups.long()), f"../data/output/three_classifiers_{run_suffix}_final_ranking.pt")

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