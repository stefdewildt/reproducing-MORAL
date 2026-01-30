# Author: F.P.J. de Kam (floris.de.kam@student.uva.nl)

import argparse
from pathlib import Path
from loguru import logger

from models.DPAH import DPAH
from main import run_single

def train_dpah(
    fm_values,
    h_MM_values,
    h_mm_values,
    N,
    d,
    plo_M,
    plo_m,
    feature_dim_DPAH,
    DPAH_seed,
    split_path_dpah,
    moral_args: argparse.Namespace,
    assume_undirected: bool,
):
    """
    Training loop for generating DPAH graphs with different parameters,
    and running the MORAL training pipeline on them.
    """
    split_dir = Path(split_path_dpah)
    for fm in fm_values:
        for h_MM in h_MM_values:
            for h_mm in h_mm_values:
                                        # create id and path
                dataset_id = "dpah_fm{:.2f}_hMM{:.2f}_hmm{:.2f}".format(fm, h_MM, h_mm)
                path = f"{dataset_id}.pt"
                                        # make DPAH graph
                Graph = DPAH(N, fm, d, plo_M, plo_m, h_MM, h_mm, verbose=False, seed=DPAH_seed)
                                        # prepare args for MORAL training
                run_args = argparse.Namespace(**vars(moral_args))
                run_args.dataset = dataset_id
                run_args.splits_dir = split_dir

                                        # call MORAL training pipeline from main.py on the in-memory synthetic graph.
                logger.info(
                    "Training MORAL on '{}' with model='{}' runs={} seed_base={}",
                    dataset_id,
                    run_args.model,
                    run_args.runs,
                    run_args.seed,
                )
                                        # Run multiple times with different seeds, 
                                        # each run will create its own split files in split_dir
                for run in range(run_args.runs):
                    run_single(
                        run_args,
                        run,
                        G=Graph,
                        feature_dim_G=feature_dim_DPAH,
                        seed_G=DPAH_seed,
                        assume_undirected=assume_undirected,
                    )

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments. Default parameters for DPAH graph generation 
    and copy of MORAL training args from main.py.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate DPAH graphs / splits and (optionally later) forward MORAL training args to main.py."
        )
    )
                                        # DPAH generation args (train_dpah params)
    dpah = parser.add_argument_group("DPAH")
    dpah.add_argument(
        "--fm_values",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        help="Fraction(s) of minority nodes (fm).",
    )
    dpah.add_argument(
        "--h_MM_values",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Homophily value(s) among majorities (h_MM).",
    )
    dpah.add_argument(
        "--h_mm_values",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Homophily value(s) among minorities (h_mm).",
    )
    dpah.add_argument("--N", type=int, default=1000, help="Number of nodes.")
    dpah.add_argument("--d", type=float, default=0.006, help="Desired edge density.")
    dpah.add_argument(
        "--plo_M",
        type=float,
        default=2.5,
        help="Power-law outdegree parameter for majority class.",
    )
    dpah.add_argument(
        "--plo_m",
        type=float,
        default=2.5,
        help="Power-law outdegree parameter for minority class.",
    )
    dpah.add_argument(
        "--feature_dim_DPAH",
        type=int,
        default=16,
        help="Feature dimensionality used when wrapping DPAH as a dataset.",
    )
    dpah.add_argument(
        "--DPAH_seed",
        type=int,
        default=42,
        help="Random seed used during DPAH graph generation.",
    )
    dpah.add_argument(
        "--split_path_dpah",
        type=Path,
        default=Path("../data/splits/dpah/"),
        help="Directory where DPAH split .pt files will be created/read.",
    )
    dpah.add_argument(
        "--assume_undirected",
        action="store_false",
        help="If set, create splits assuming the graph is undirected.",
    )

                                        # MORAL args mirrors main.py, kept for later forwarding
    moral = parser.add_argument_group("MORAL (mirrors main.py)")
    moral.add_argument(
        "--model",
        type=str,
        default="gae",
        choices={"gae", "ncn"},
        help="Base encoder/decoder setup.",
    )
    moral.add_argument(
        "--fair_model",
        type=str,
        default="moral",
        help="Name of the fairness method (for logging only).",
    )
    moral.add_argument("--device", type=str, default="cpu", help="PyTorch device string.")
    moral.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    moral.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Training batch size per sensitive group.",
    )
    moral.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    moral.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimensionality of the encoders.",
    )
    moral.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay used by Adam.")
    moral.add_argument("--runs", type=int, default=3, help="Number of runs with different seeds.")
    moral.add_argument("--seed", type=int, default=0, help="Base seed used for reproducibility.")
    moral.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("../data/splits/dpah/"),
        help="Directory containing pre-computed edge splits (train/valid/test). Overwritten by DPAH splits.",
    )
    moral.add_argument(
        "--encoder",
        type=str,
        choices=["gcn", "gin", "sage"],
        default=None,
        help="Override encoder type (e.g. 'gcn', 'sage', 'gin').  "
            "If not set, uses encoder from --model config.",
    )
    moral.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Count of X many layers for the encoder (default: 2).",
    )
    moral.add_argument(
        "--ranking_loss",
        action="store_true",
        help="Deprecated option kept for compatibility with older scripts.",
    )
    moral.add_argument(
        "--baseline",
        action="store_true",
        help="Deprecated option kept for compatibility with older scripts.",
    )
    moral.add_argument(
        "--csv_out",
        type=Path,
        default=Path("../data/output/dpah/dpah_edge_distributions.csv"),
        help="CSV file to store graph + split statistics (upserted).",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logger.info(
        "Generating DPAH graphs: fm_values={} h_MM_values={} h_mm_values={} (N={}, d={})",
        args.fm_values,
        args.h_MM_values,
        args.h_mm_values,
        args.N,
        args.d,
    )

    train_dpah(
        fm_values=args.fm_values,
        h_MM_values=args.h_MM_values,
        h_mm_values=args.h_mm_values,
        N=args.N,
        d=args.d,
        plo_M=args.plo_M,
        plo_m=args.plo_m,
        feature_dim_DPAH=args.feature_dim_DPAH,
        DPAH_seed=args.DPAH_seed,
        split_path_dpah=args.split_path_dpah,
        moral_args=args,
        assume_undirected=args.assume_undirected,
    )


if __name__ == "__main__":
    main()