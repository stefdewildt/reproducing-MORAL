# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fair Adjacency Graph Embedding for Link Prediction")

    # experiment
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--runs', type=int, default=3, help='number of runs')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--dataset', type=str, default="facebook")
    parser.add_argument('--scale', action="store_false", help='normalize the data')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='proportion of testing edges')
    parser.add_argument('--splits_dir', type=str, default="../data/splits", help='path to the specific data split') #might change in snellius
    parser.add_argument('--csv_out', type=str, default="../data/output/fairadj_edge_distributions.csv", help='path to save edge distribution csv')

    # model
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

    # optimize
    parser.add_argument('--outer_epochs', type=int, default=4, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument("--eta", type=float, default=None, help="Fairness trade-off (T2 loop). Auto-selected per dataset if None.")
    parser.add_argument("--T1", type=int, default=50, help="Utility training epochs")
    parser.add_argument("--T2", type=int, default=20, help="Adjacency optimization epochs")
    parser.add_argument('--eq', action="store_true", help='Set to true for Oklahoma97 and UNC28')

    # for adversarial
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--alpha', type=float, default=10., help='hyperparameter for adversarial loss')
    parser.add_argument('--lr_mult', type=float, default=1., help='learning rate multiple for adversarial net')

    return parser.parse_args()