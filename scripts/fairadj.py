# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import os
import scipy.sparse as sp
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from loguru import logger

from baseline.fairadj.fairadj_helpers.args import parse_args
from baseline.fairadj.fairadj_helpers.utils import fix_seed, find_link
from baseline.fairadj.utils import preprocess_graph, project
from baseline.fairadj.optimizer import loss_function
from baseline.fairadj.gae import GCNModelVAE

from helpers.utils import moral_aggregate_algorithm, get_dataset
from helpers.check_splits import main as check_splits
from helpers.create_splits import canonicalize_undirected_edges


def get_dataset_eta(dataset_name: str) -> float:
    """Return dataset-specific eta based on graph characteristics."""
    eta_map = {
        'cora': 0.2,
        'citeseer': 5.0,
        'facebook': 0.2,
        'german': 0.05,
        'nba': 0.02,
        'pokec_n': 3.0,
        'pokec_z': 2.5,
        'credit': 0.15,
        'chameleon': 0.3,
        'airtraffic': 0.5,
    }
    return eta_map.get(dataset_name.lower(), 0.2)


def main(args):
# Stap 1: Laad de data en pak de tuple uit (PyTorch 2.6 fix)
    # Auto-select eta if not specified
    if args.eta is None:
        args.eta = get_dataset_eta(args.dataset)
        logger.info(f"Auto-selected eta={args.eta} for dataset {args.dataset}")
    
    for run in range(args.runs):

        logger.info(f"Run {run + 1}/{args.runs} with seed {args.seed + run}")
        seed = args.seed + run
        fix_seed(seed)

        #1. Load data
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data_obj, splits = get_dataset(
            args.dataset,
            args.splits_dir,
            seed,
        )

        # Stap 2: Bouw het pad naar het split-bestand met os.path.join
        # Dit vervangt de "/" die de error veroorzaakte
        filename = f"{args.dataset}_{seed}.pt"
        current_split_file = os.path.join(args.splits_dir, filename)
        
        # Statistieken loggen (gebruik str() voor de zekerheid)
        check_splits(current_split_file, csv_out=str(args.csv_out), verbose=False)

        #2. prepare variable for FairAdj
        features = features.to(args.device)
        sens_vec = sens.long()
        n_nodes, feat_dim = features.shape

        edge_index = data_obj.edge_index
        adj_sparse = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),
            shape=(n_nodes, n_nodes)
        )

        #3. preprocess normalized adj matrices
        adj_norm = preprocess_graph(adj_sparse).to(args.device)
        adj_label_mat = sp.coo_matrix(adj_sparse + sp.eye(adj_sparse.shape[0]))
        adj_label = torch.FloatTensor(adj_label_mat.toarray()).to(args.device)

        #4. FairAdj specific variables
        sensivive_np = sens_vec.cpu().numpy()
        intra_pos, inter_pos, intra_link_pos, inter_link_pos = find_link(adj_sparse, sensivive_np)

        pos_weight = float(adj_sparse.shape[0] * adj_sparse.shape[0] - adj_sparse.sum()) / adj_sparse.sum()
        pos_weight = torch.tensor(pos_weight).to(args.device)
        norm = adj_sparse.shape[0] * adj_sparse.shape[0] / float((adj_sparse.shape[0] * adj_sparse.shape[0] - adj_sparse.sum()) * 2)

        # Initialization
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(args.device)
        optimizer = optim.Adam(model.get_parameters(), lr=args.lr)

        # Training
        model.train()
        for i in range(args.outer_epochs):
            for epoch in range(args.T1):
                optimizer.zero_grad()

                recovered, z, mu, logvar = model(features, adj_norm)
                loss = loss_function(preds=recovered, labels=adj_label, mu=mu, logvar=logvar, n_nodes=n_nodes, norm=norm,
                                    pos_weight=pos_weight)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                cur_loss = loss.item()
                
                if not torch.isfinite(torch.tensor(cur_loss)):
                    print(f"Epoch in T1: [{epoch + 1}/{args.T1}]; Loss: nan (skipped)")
                    optimizer.zero_grad()
                    continue
                    
                optimizer.step()

                print("Epoch in T1: [{:d}/{:d}];".format((epoch + 1), args.T1), "Loss: {:.3f};".format(cur_loss))

            for epoch in range(args.T2):
                adj_norm = adj_norm.requires_grad_(True)
                recovered = model(features, adj_norm)[0]

                if args.eq:
                    intra_score = recovered[intra_link_pos[:, 0], intra_link_pos[:, 1]].mean()
                    inter_score = recovered[inter_link_pos[:, 0], inter_link_pos[:, 1]].mean()
                else:
                    intra_score = recovered[intra_pos[:, 0], intra_pos[:, 1]].mean()
                    inter_score = recovered[inter_pos[:, 0], inter_pos[:, 1]].mean()

                loss = F.mse_loss(intra_score, inter_score)
                loss.backward()
                cur_loss = loss.item()
                
                if not torch.isfinite(torch.tensor(cur_loss)) or (adj_norm.grad is not None and torch.isnan(adj_norm.grad).any()):
                    print(f"Epoch in T2: [{epoch + 1}/{args.T2}]; Loss: {cur_loss} (skipped due to NaN)")
                    adj_norm = adj_norm.detach()
                    continue

                print("Epoch in T2: [{:d}/{:d}];".format(epoch + 1, args.T2), "Loss: {:.5f};".format(cur_loss))

                # Original update (same as FairAdj repo)
                adj_norm = adj_norm.add(adj_norm.grad.mul(-args.eta)).detach()
                adj_norm = adj_norm.to_dense()

                for i in range(adj_norm.shape[0]):
                    adj_norm[i] = project(adj_norm[i])

                adj_norm = adj_norm.to_sparse()

        # Evaluation
        model.eval()
        with torch.no_grad():
            z = model(features, adj_norm)[1]
        
        # print(f"Final ranking data saved to: {save_path}")
        edge_index_canon = canonicalize_undirected_edges(data_obj.edge_index)
        u, v = edge_index_canon[0], edge_index_canon[1]
        # groups: inter, intra
        global_edge_groups = sens_vec[u] + sens_vec[v]
        pi_t = torch.bincount(global_edge_groups, minlength=3).float()
        pi = pi_t / pi_t.sum()

        test_split = splits['test']
        test_edges = torch.cat([test_split['edge'], test_split['edge_neg']], dim=0).long()
        test_labels = torch.cat([
            torch.ones(test_split['edge'].shape[0]),
            torch.zeros(test_split['edge_neg'].shape[0])
        ], dim=0)

        outputs = recovered[test_edges[:, 0], test_edges[:, 1]].cpu()
        edge_sens_groups = (sens_vec[test_edges[:, 0]] + sens_vec[test_edges[:, 1]]).long()

        run_suffix = f"{args.dataset}_fairadj_GAE_{seed}" 

        torch.save((outputs, edge_sens_groups), f"../data/output/three_classifiers_{run_suffix}.pt")

        K = 1000
        final_output, final_labels, final_groups = moral_aggregate_algorithm(
            outputs=outputs,
            test_labels=test_labels,
            edge_sens_groups=edge_sens_groups,
            pi=pi,
            K=K,
        )

        # Create output_positions to match format: (final_output, final_labels, output_positions, final_groups)
        output_positions = torch.arange(len(final_output), dtype=torch.long)
        
        torch.save(
            (final_output, final_labels, output_positions, final_groups),
            f"../data/output/three_classifiers_{run_suffix}_final_ranking.pt"
        )

        print(f"FairAdj results saved for moral ranking.")

    return


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device(args.device)
    fix_seed(args.seed)
    main(args)