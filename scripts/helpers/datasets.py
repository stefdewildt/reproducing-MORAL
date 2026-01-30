# Author: Mattos et al., modified by F.P.J. de Kam (floris.de.kam@student.uva.nl)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os
import networkx as nx
import requests
import os
import zipfile
import io
import gdown
import requests
import random
from itertools import combinations
import networkx as nx
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


################################################################
# General Dataset class (original code)
################################################################

def feature_norm(self, features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    denom = (max_values - min_values)
    denom[denom == 0] = 1  # avoid division by zero
    return 2 * (features - min_values).div(denom) - 1

class Dataset(object):
    def __init__(self, is_normalize: bool = False, root: str = "../../dataset") -> None:
        self.adj_ = None
        self.features_ = None
        self.labels_ = None
        self.idx_train_ = None
        self.idx_val_ = None
        self.idx_test_ = None
        self.sens_ = None
        self.sens_idx_ = None
        self.is_normalize = is_normalize
    
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.path_name = ""

    def download(self, url: str, filename: str):
        r = requests.get(url)
        assert r.status_code == 200
        open(os.path.join(self.root, self.path_name, filename), "wb").write(r.content)

    def download_zip(self, url: str):
        r = requests.get(url)
        assert r.status_code == 200
        foofile = zipfile.ZipFile(io.BytesIO(r.content))
        foofile.extractall(os.path.join(self.root, self.path_name))

    def adj(self, datatype: str = "torch.sparse"):
        # assert str(type(self.adj_)) == "<class 'torch.Tensor'>"
        if self.adj_ is None:
            return self.adj_
        if datatype == "torch.sparse":
            return self.adj_
        elif datatype == "scipy.sparse":
            return sp.coo_matrix(self.adj.to_dense())
        elif datatype == "np.array":
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError(
                "datatype should be torch.sparse, tf.sparse, np.array, or scipy.sparse"
            )

    def features(self, datatype: str = "torch.tensor"):
        if self.is_normalize and self.features_ is not None:
            self.features_ = feature_norm(self, self.features_)

        if self.features is None:
            return self.features_
        if datatype == "torch.tensor":
            return self.features_
        elif datatype == "np.array":
            return self.features_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def labels(self, datatype: str = "torch.tensor"):
        if self.labels_ is None:
            return self.labels_
        if datatype == "torch.tensor":
            return self.labels_
        elif datatype == "np.array":
            return self.labels_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_val(self, datatype: str = "torch.tensor"):
        if self.idx_val_ is None:
            return self.idx_val_
        if datatype == "torch.tensor":
            return self.idx_val_
        elif datatype == "np.array":
            return self.idx_val_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_train(self, datatype: str = "torch.tensor"):
        if self.idx_train_ is None:
            return self.idx_train_
        if datatype == "torch.tensor":
            return self.idx_train_
        elif datatype == "np.array":
            return self.idx_train_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_test(self, datatype: str = "torch.tensor"):
        if self.idx_test_ is None:
            return self.idx_test_
        if datatype == "torch.tensor":
            return self.idx_test_
        elif datatype == "np.array":
            return self.idx_test_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens(self, datatype: str = "torch.tensor"):
        if self.sens_ is None:
            return self.sens_
        if datatype == "torch.tensor":
            return self.sens_
        elif datatype == "np.array":
            return self.sens_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens_idx(self):
        if self.sens_idx_ is None:
            self.sens_idx_ = -1
        return self.sens_idx_


################################################################
# Our contribution
################################################################

def edge_homophily_binary(edge_index: np.ndarray, sens: np.ndarray) -> float:
    """
    Fraction of edges whose endpoints share the same binary sensitive label.
    """
    u, v = edge_index
    same = (sens[u] == sens[v]).sum()
    return float(same / edge_index.shape[1])


def find_most_heterophilic_split(edge_index: np.ndarray, labels_5: np.ndarray):
    """
    Return the binary split of classes that minimizes edge homophily.
    """
    labels_5 = np.asarray(labels_5, dtype=int)
    classes = tuple(sorted(set(labels_5.tolist())))
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes present to form a binary split.")
                                        # count instances per class
    counts = np.bincount(labels_5, minlength=5).astype(int)
    present_classes = [c for c in range(5) if counts[c] > 0]

    best = None
    best_groupA = None
                                        # try all possible non-trivial splits
    for r in range(1, len(present_classes)):
        for subset in combinations(present_classes, r):
            A = set(subset)
            B = set(present_classes) - A
            nA = sum(counts[c] for c in A)
            nB = sum(counts[c] for c in B)
            
            if nA == 0 or nB == 0:
                continue
                                        # compute sens array for this split
            sens = np.isin(labels_5, list(A)).astype(np.int8)
            h = edge_homophily_binary(edge_index=edge_index, sens=sens)

            if best is None or h < best:
                best = h
                best_sens = sens
                best_groupA = tuple(sorted(A))

    if best is None:
        raise RuntimeError("No valid heterophilic split found.")
                                        # return sens array, classes in group 1, homophily
    return best_sens, best_groupA, best  

class Chameleon(Dataset):
    def __init__(
        self, 
        return_tensor_sparse: bool = True, 
        is_normalize: bool = False, 
        root: str = "../dataset", 
        seed: int = 42
    ):
        super().__init__() 
        self.root = root
        self.path_name = "chameleon"
        self.seed = seed

                                        # load data using the helper method
        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
        ) = self.load_chameleon(seed=self.seed)

                                        # convert adjacency to torch sparse tensor
        self.adj_ = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )
        
        self.features_ = features
        self.labels_ = labels
        self.sens_ = sens
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_idx_ = -1

    def load_chameleon(self, seed):
        """Downloads and processes the Chameleon filtered dataset"""
                                        # ensure the directory exists
        data_dir = os.path.join(self.root, self.path_name)
        os.makedirs(data_dir, exist_ok=True)

        url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/chameleon_filtered.npz"
        filename = "chameleon_filtered.npz"
        npz_path = os.path.join(data_dir, filename)

                                        # download if file is missing
        if not os.path.exists(npz_path):
            print(f"Downloading {filename}...")
            self.download(url, filename)

                                        # load the .npz file
        data = np.load(npz_path, allow_pickle=True)
        x = data['node_features'].astype(np.float32)
        y = data['node_labels'].astype(np.int64)
        edges = data['edges'].astype(np.int64)

        train_mask = data['train_masks'].astype(bool)
        val_mask = data['val_masks'].astype(bool)
        test_mask = data['test_masks'].astype(bool)

                                        # handle 10-split structure, 
                                        # take first split to maintain consistency
        if len(train_mask.shape) > 1:
            train_mask = train_mask[:, 0]
            val_mask = val_mask[:, 0]
            test_mask = test_mask[:, 0]

        num_nodes = x.shape[0]

                                        # build adjacency matrix (sparse, undirected)
        row, col = edges[:, 0], edges[:, 1]
        adj = sp.coo_matrix((np.ones(edges.shape[0], dtype=np.float32), (row, col)),
                            shape=(num_nodes, num_nodes), dtype=np.float32)
        
                                        # ensure it is undirected
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        
                                        # binary sensitive attribute (lowest homophily split)
        edge_index = np.vstack([row, col])
        sens, groupA_classes, homophily = find_most_heterophilic_split(edge_index=edge_index, labels_5=y)

                                        # convert to torch tensors
        features = torch.FloatTensor(x)
        labels = torch.LongTensor(y)
        sens = torch.LongTensor(sens)

        idx_train = torch.from_numpy(np.where(train_mask)[0]).long()
        idx_val = torch.from_numpy(np.where(val_mask)[0]).long()
        idx_test = torch.from_numpy(np.where(test_mask)[0]).long()

                                        # append sensitive attribute as last feature
        features = torch.cat([features, sens.unsqueeze(-1).float()], dim=1)

        return adj, features, labels, idx_train, idx_val, idx_test, sens



def map_education_to_3class(values) -> np.ndarray:
    """
    Map EducationLevel to {0,1,2}. If there are exactly 3 unique values: factorize to 0..2
    Otherwise (UCI-style): 1/2 -> 0, 3 -> 1, rest -> 2. Fallback: quantile binning into 3 buckets
    """
    import numpy as np
    import pandas as pd

                                        # robust: coerce naar numeric, NaN -> -1
    v = pd.to_numeric(pd.Series(values), errors="coerce").fillna(-1).astype(int).to_numpy()
    uniq = np.unique(v)

                                        # if already 0,1,2 or subset thereof, return as is
    if set(uniq).issubset({0, 1, 2}) and len(uniq) <= 3:
        return v

                                        # exactly 3 unique values: factorize to 0,1,2
    if len(uniq) == 3:
        mapping = {u: i for i, u in enumerate(sorted(uniq))}
        return np.vectorize(mapping.get)(v).astype(int)

                                        # UCI credit education style: 1=grad,2=uni,3=hs,4+=other/unknown
    mapped = np.full_like(v, 2)
    mapped[np.isin(v, [1, 2])] = 0
    mapped[v == 3] = 1

    if len(np.unique(mapped)) == 3:
        return mapped

                                        # fallback: make 3 quantile bins
    bins = pd.qcut(v, 3, labels=False, duplicates="drop")
    bins = np.asarray(bins).astype(int)

    if len(np.unique(bins)) < 3:
        uniq2 = np.unique(v)
        mapping = {u: min(i, 2) for i, u in enumerate(sorted(uniq2))}
        return np.vectorize(mapping.get)(v).astype(int)

    return bins


class CreditMultiClass(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "../dataset"):
        super().__init__(is_normalize=is_normalize, root=root)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_credit_multiclass("credit_multiclass")

                                        # convert to tensors
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        labels = labels.long()

                                        # convert scipy sparse adjacency to torch sparse tensor
        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)

        self.adj_ = adj
        self.features_ = features       # torch.FloatTensor
        self.labels_ = labels           # torch.LongTensor
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens.long()        # LongTensor with 3 classes
        self.sens_idx_ = sens_idx              

    def load_credit_multiclass(
        self,
        dataset: str,
        sens_attr: str = "EducationLevel",
        predict_attr: str = "NoDefaultNextMonth",
        path: str = "../dataset/credit_multiclass/",
        label_number: int = 6000,
    ):
                                        # downloads identical to Credit.load_credit(...)
        self.path_name = "credit_multiclass"
        base_dir = os.path.join(self.root, self.path_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        csv_path = os.path.join(base_dir, "credit_multiclass.csv")
        edge_path = os.path.join(base_dir, "credit_multiclass_edges.txt")

        if not os.path.exists(csv_path):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/credit/credit.csv"
            self.download(url, "credit_multiclass.csv")
        if not os.path.exists(edge_path):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/credit/credit_edges.txt"
            self.download(url, "credit_multiclass_edges.txt")

        idx_features_labels = pd.read_csv(os.path.join(base_dir, f"{dataset}.csv"))

        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        if "Single" in header:
            header.remove("Single")

                                        # remove sens_attr from features to prevent leakage
        if sens_attr in header:
            header.remove(sens_attr)

        edges_unordered = np.genfromtxt(
            os.path.join(base_dir, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)

                                        # label as numpy for splitting
        labels_np = idx_features_labels[predict_attr].to_numpy()

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels_np.shape[0], labels_np.shape[0]),
            dtype=np.float32,
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

                                        # convert to torch tensors
        features = torch.FloatTensor(features.toarray())
        labels = torch.LongTensor(labels_np)

        random.seed(20)
        label_idx_0 = np.where(labels_np == 0)[0]
        label_idx_1 = np.where(labels_np == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens_raw = idx_features_labels[sens_attr].to_numpy()
        sens3 = map_education_to_3class(sens_raw)
        sens = torch.LongTensor(sens3)

        sens_idx = idx_features_labels.columns.get_loc(sens_attr)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, sens_idx
    
def mx_to_torch_sparse_tensor(
    sparse_mx, 
    is_sparse=False, 
    return_tensor_sparse=True
):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    if not is_sparse:
        sparse_mx = sp.coo_matrix(sparse_mx)
    else:
        sparse_mx = sparse_mx.tocoo()
    if not return_tensor_sparse:
        return sparse_mx

    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class DPAH_dataset(Dataset):
    def __init__(
        self,
        G: nx.DiGraph,
        feature_dim: int = 16,
        is_normalize: bool = False,
        root: str = "../dataset",
        seed: int = 42,
    ) -> None:
        """
        Wrap an in-memory DPAH NetworkX graph as a Dataset.
        Features are i.i.d. Gaussian noise (N(0,1)) of shape [N, feature_dim].
        Sensitive attribute: are node binary minority/majority label (from node attribute).
        They are NOT appended to features.
        Labels are trivial zeros (some pipeline code expects y to exist).
        """

        super().__init__(is_normalize=is_normalize, root=root)
        self.path_name = "dpah"

        if G is None:
            raise ValueError("DPAH Dataset requires a NetworkX graph G (got None).")

                                        # node ordering + index mapping
        node_list = list(G.nodes())
        if len(node_list) == 0:
            raise ValueError("Graph G has no nodes.")

        try:
            node_list = sorted(node_list)
        except TypeError:
                                        # mixed / non-orderable node types
            node_list = list(node_list)

        node_to_idx = {node: i for i, node in enumerate(node_list)}
        num_nodes = len(node_list)

                                        # sensitive attribute key
        sens_key = None
        if hasattr(G, "graph") and isinstance(G.graph, dict):
            sens_key = G.graph.get("label")
        if not sens_key:
            sens_key = "m"              # default used in DPAH.py

        sens = np.empty(num_nodes, dtype=np.int64)
        for node in node_list:
            attrs = G.nodes[node]
            if sens_key in attrs:
                sens[node_to_idx[node]] = int(attrs[sens_key])
            elif "sens" in attrs:
                sens[node_to_idx[node]] = int(attrs["sens"])
            else:
                raise ValueError(
                    f"Node '{node}' missing sensitive attribute '{sens_key}'. "
                    "Expected DPAH-style node attributes (e.g. {'m': 0/1})."
                )

                                        # random Gaussian node features,
                                        # do not include sensitive attribute
        rng_feat = np.random.default_rng(int(seed))
        features = rng_feat.standard_normal((num_nodes, int(feature_dim)), dtype=np.float32)

                                        # trivial labels, kept for compatibility with code expecting y
        labels = np.zeros(num_nodes, dtype=np.int64)

                                        # node-level train/val/test splits,
                                        # this is different from the edge-level splits made later
        rng = np.random.default_rng(int(seed))
        idx = np.arange(num_nodes)
        rng.shuffle(idx)
        n_train = int(0.8 * num_nodes)
        n_val = int(0.1 * num_nodes)
        idx_train = idx[:n_train]
        idx_val = idx[n_train : n_train + n_val]
        idx_test = idx[n_train + n_val :]

                                        # build adjacency, directed if G is directed
        edges = [(node_to_idx[u], node_to_idx[v]) for (u, v) in G.edges()]
        if len(edges) == 0:
            adj = sp.coo_matrix((num_nodes, num_nodes), dtype=np.float32)
        else:
            rows = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
            cols = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
            data = np.ones(len(edges), dtype=np.float32)
            adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)

                                        # convert to torch tensors
        self.features_ = torch.FloatTensor(features)
        self.labels_ = torch.LongTensor(labels)
        self.sens_ = torch.LongTensor(sens)

        self.idx_train_ = torch.LongTensor(idx_train)
        self.idx_val_ = torch.LongTensor(idx_val)
        self.idx_test_ = torch.LongTensor(idx_test)

        self.adj_ = mx_to_torch_sparse_tensor(adj, is_sparse=True)

                                        # even though most code relies on data.sens, set sens_idx_ for compatibility
        self.sens_idx_ = -1


class AirTraffic(Dataset):
    def __init__(
        self,
        hub_percentile: float = 90.0,
        return_tensor_sparse: bool = True,
        is_normalize: bool = False,
        root: str = "../dataset",
        seed: int = 42,
    ):
        super().__init__(is_normalize=is_normalize, root=root)
        self.path_name = "airtraffic"
        self.hub_percentile = hub_percentile
        self.seed = seed

                                        # load data using the helper method
        adj, features, labels, idx_train, idx_val, idx_test, sens = self.load_airtraffic(seed=self.seed)
                                        # convert adjacency to torch sparse tensor
        self.adj_ = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )
        
        self.features_ = features
        self.labels_ = labels
        self.sens_ = sens
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_idx_ = -1             # sensitive attribute is already appended to features

    def load_airtraffic(self, seed):
        """
        Downloads raw OpenFlights data and constructs the graph objects
        """
        data_dir = os.path.join(self.root, self.path_name)
        os.makedirs(data_dir, exist_ok=True)

                                        # direct links to raw OpenFlights data from GitHub mirror
        nodes_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
        edges_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

        nodes_path = os.path.join(data_dir, "airports.csv")
        edges_path = os.path.join(data_dir, "routes.csv")

                                        # download files if they do not exist locally
        if not os.path.exists(nodes_path):
            self.download(nodes_url, "airports.csv")
        if not os.path.exists(edges_path):
            self.download(edges_url, "routes.csv")

                                        # process nodes (Airports)
        node_cols = [
            "airport_id", "name", "city", "country", "iata", "icao",
            "latitude", "longitude", "altitude", "timezone", "dst",
            "tz_type", "type", "source"
        ]
                                        # OpenFlights .dat files use \N for NULL values
        nodes_df = pd.read_csv(nodes_path, header=None, names=node_cols, na_values="\\N")
        
                                        # ensure unique integer IDs,
                                        # create a mapping to 0-indexed range
        nodes_df = nodes_df.dropna(subset=['airport_id'])
        nodes_df['airport_id'] = nodes_df['airport_id'].astype(int)
        id_map = {old_id: i for i, old_id in enumerate(nodes_df['airport_id'])}
        num_nodes = len(nodes_df)

                                        # process edges (Routes)
        edge_cols = [
            "airline", "airline_id", "source_airport", "source_airport_id",
            "target_airport", "target_airport_id", "codeshare", "stops", "equipment"
        ]
        edges_df = pd.read_csv(edges_path, header=None, names=edge_cols, na_values="\\N")
        
                                        # filter out edges where airport IDs are missing or not in our node list
        edges_df = edges_df.dropna(subset=['source_airport_id', 'target_airport_id'])
        edges_df['source_airport_id'] = edges_df['source_airport_id'].astype(int)
        edges_df['target_airport_id'] = edges_df['target_airport_id'].astype(int)
        
        mask = edges_df['source_airport_id'].isin(id_map) & edges_df['target_airport_id'].isin(id_map)
        valid_edges = edges_df[mask]

                                        # map original IDs to our 0-indexed integer IDs
        src = valid_edges['source_airport_id'].map(id_map).values
        tgt = valid_edges['target_airport_id'].map(id_map).values

                                        # construct adjacency matrix
        adj = sp.coo_matrix(
            (np.ones(len(src)), (src, tgt)),
            shape=(num_nodes, num_nodes),
            dtype=np.float32,
        )
        
                                        # make the graph undirected and remove duplicates/self-loops
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

                                        # process node features (latitude, longitude, altitude, timezone)
        feature_cols = ["latitude", "longitude", "altitude", "timezone"]
        features_np = nodes_df[feature_cols].fillna(0).astype(np.float32).values

                                        # generate labels and sensitive attributes (degree-based)
        degree = np.asarray(adj.sum(axis=1)).flatten()
        
                                        # sensitive attribute: Hub (1) vs Non-hub (0) based on percentile
        hub_threshold = np.percentile(degree, self.hub_percentile)
        sens_np = (degree >= hub_threshold).astype(np.int64)
        
                                        # labels: High-degree (1) vs Low-degree (0) based on median split
        label_threshold = np.median(degree)
        labels_np = (degree >= label_threshold).astype(np.int64)

                                        # train / val / test split for nodes (not edges)
        rng = np.random.default_rng(seed)
        idx = np.arange(num_nodes)
        rng.shuffle(idx)

        n_train = int(0.8 * num_nodes)
        n_val = int(0.1 * num_nodes)

        idx_train = torch.LongTensor(idx[:n_train])
        idx_val = torch.LongTensor(idx[n_train : n_train + n_val])
        idx_test = torch.LongTensor(idx[n_train + n_val :])

                                        # tensor conversion
        features = torch.FloatTensor(features_np)
        labels = torch.LongTensor(labels_np)
        sens = torch.LongTensor(sens_np)

                                        # append sensitive attribute as the last feature column
        features = torch.cat([features, sens.unsqueeze(-1).float()], dim=1)

        return adj, features, labels, idx_train, idx_val, idx_test, sens


################################################################
# Original code
################################################################

class Facebook(Dataset):
    def __init__(
        self,
        path: str = "../dataset/facebook/",
        is_normalize: bool = False,
        root: str = "../dataset",
    ) -> None:
        super().__init__(is_normalize=is_normalize, root=root)
        self.path_name = "facebook"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(os.path.join(self.root, self.path_name, "107.edges")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/facebook/facebook/107.edges"
            filename = "107.edges"
            self.download(url, filename)
        if not os.path.exists(os.path.join(self.root, self.path_name, "107.feat")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/facebook/facebook/107.feat"
            filename = "107.feat"
            self.download(url, filename)
        if not os.path.exists(os.path.join(self.root, self.path_name, "107.featnames")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/facebook/facebook/107.featnames"
            filename = "107.featnames"
            self.download(url, filename)

        edges_file = open(os.path.join(self.root, self.path_name, "107.edges"))
        edges = []
        for line in edges_file:
            edges.append([int(one) for one in line.strip("\n").split(" ")])

        feat_file = open(os.path.join(self.root, self.path_name, "107.feat"))
        feats = []
        for line in feat_file:
            feats.append([int(one) for one in line.strip("\n").split(" ")])

        feat_name_file = open(os.path.join(self.root, self.path_name, "107.featnames"))
        feat_name = []
        for line in feat_name_file:
            feat_name.append(line.strip("\n").split(" "))
        names = {}
        for name in feat_name:
            if name[1] not in names:
                names[name[1]] = name[1]

        feats = np.array(feats)

        node_mapping = {}
        for j in range(feats.shape[0]):
            node_mapping[feats[j][0]] = j

        feats = feats[:, 1:]

        sens = feats[:, 264]
        labels = feats[:, 220]

        feats = np.concatenate([feats[:, :264], feats[:, 266:]], -1)

        feats = np.concatenate([feats[:, :220], feats[:, 221:]], -1)

        edges = np.array(edges)
        # edges=torch.tensor(edges)
        # edges=torch.stack([torch.tensor(one) for one in edges],0)

        node_num = feats.shape[0]
        adj = np.zeros([node_num, node_num])

        for j in range(edges.shape[0]):
            adj[node_mapping[edges[j][0]], node_mapping[edges[j][1]]] = 1

        idx_train = np.random.choice(
            list(range(node_num)), int(0.8 * node_num), replace=False
        )
        idx_val = list(set(list(range(node_num))) - set(idx_train))
        idx_test = np.random.choice(idx_val, len(idx_val) // 2, replace=False)
        idx_val = list(set(idx_val) - set(idx_test))

        self.features_ = torch.FloatTensor(feats)
        self.sens_ = torch.FloatTensor(sens)
        self.idx_train_ = torch.LongTensor(idx_train)
        self.idx_val_ = torch.LongTensor(idx_val)
        self.idx_test_ = torch.LongTensor(idx_test)
        self.labels_ = torch.LongTensor(labels)

        self.features_ = torch.cat([self.features_, self.sens_.unsqueeze(-1)], -1)
        self.adj_ = mx_to_torch_sparse_tensor(adj)
        self.sens_idx_ = -1


class Nba(Dataset):
    def __init__(
        self,
        dataset_name="nba",
        predict_attr_specify=None,
        return_tensor_sparse=True,
        is_normalize: bool = False,
        root: str = "../dataset",
    ):
        super().__init__(is_normalize=is_normalize, root=root)
        if dataset_name != "nba":
            if dataset_name == "pokec_z":
                dataset = "region_job"
            elif dataset_name == "pokec_n":
                dataset = "region_job_2"
            else:
                dataset = None
            sens_attr = "gender"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path = "../dataset/pokec/"
            test_idx = False
        else:
            dataset = "nba"
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "../dataset/NBA"
            test_idx = True

        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
            idx_sens_train,
        ) = self.load_pokec(
            dataset,
            sens_attr,
            predict_attr if predict_attr_specify == None else predict_attr_specify,
            path=path,
            label_number=label_number,
            sens_number=sens_number,
            seed=seed,
            test_idx=test_idx,
        )

        # adj=adj.todense()
        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )
        labels[labels > 1] = 1

        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = -1

    def load_pokec(
        self,
        dataset,
        sens_attr,
        predict_attr,
        path=".../dataset/pokec/",
        label_number=1000,
        sens_number=500,
        seed=19,
        test_idx=False,
    ):
        """Load data"""

        self.path_name = "nba"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(os.path.join(self.root, self.path_name, "nba.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/NBA/nba.csv"
            filename = "nba.csv"
            self.download(url, filename)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "nba_relationship.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/NBA/nba_relationship.txt"
            filename = "nba_relationship.txt"
            self.download(url, filename)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "nba.csv")
        )
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, "nba_relationship.txt"),
            dtype=np.int64,
        )

        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random

        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)) :]

        sens = idx_features_labels[sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


class Pokec_z(Dataset):
    def __init__(
        self,
        dataset_name="pokec_z",
        predict_attr_specify=None,
        return_tensor_sparse=True,
        is_normalize: bool = False,
        root: str = "../dataset",
    ):
        super().__init__(is_normalize=is_normalize, root=root)
        if dataset_name != "nba":
            if dataset_name == "pokec_z":
                dataset = "region_job"
            elif dataset_name == "pokec_n":
                dataset = "region_job_2"
            else:
                dataset = None
            sens_attr = "gender"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path = "../dataset/pokec/"
            test_idx = False
        else:
            dataset = "nba"
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "../dataset/NBA"
            test_idx = True

        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
            idx_sens_train,
        ) = self.load_pokec(
            dataset,
            sens_attr,
            predict_attr if predict_attr_specify == None else predict_attr_specify,
            path=path,
            label_number=label_number,
            sens_number=sens_number,
            seed=seed,
            test_idx=test_idx,
        )

        # adj=adj.todense(
        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )
        labels[labels > 1] = 1
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = -1

    def load_pokec(
        self,
        dataset,
        sens_attr,
        predict_attr,
        path=".../dataset/pokec/",
        label_number=1000,
        sens_number=500,
        seed=19,
        test_idx=False,
    ):
        """Load data"""

        self.path_name = "pokec_z"
        self.url = "https://drive.google.com/u/0/uc?id=1FOYOIdFp6lI9LH5FJAzLhjFCMAxT6wb4&export=download"
        self.destination = os.path.join(self.root, self.path_name, "pokec_z.zip")
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "region_job.csv")
        ):
            gdown.download(self.url, self.destination)
            with zipfile.ZipFile(self.destination, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "region_job_relationship.txt")
        ):
            gdown.download(self.url, self.destination, quiet=False)
            with zipfile.ZipFile(self.destination, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, self.path_name))

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "region_job.csv")
        )
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, "region_job_relationship.txt"),
            dtype=np.int64,
        )

        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random

        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)) :]

        sens = idx_features_labels[sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


class Pokec_n(Dataset):
    def __init__(
        self,
        dataset_name="pokec_n",
        predict_attr_specify=None,
        return_tensor_sparse=True,
        is_normalize: bool = False,
        root: str = "../dataset",
    ):
        super().__init__(is_normalize=is_normalize, root=root)
        if dataset_name != "nba":
            if dataset_name == "pokec_z":
                dataset = "region_job"
            elif dataset_name == "pokec_n":
                dataset = "region_job_2"
            else:
                dataset = None
            sens_attr = "gender"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path = "../dataset/pokec/"
            test_idx = False
        else:
            dataset = "nba"
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "../dataset/NBA"
            test_idx = True

        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
            idx_sens_train,
        ) = self.load_pokec(
            dataset,
            sens_attr,
            predict_attr if predict_attr_specify == None else predict_attr_specify,
            path=path,
            label_number=label_number,
            sens_number=sens_number,
            seed=seed,
            test_idx=test_idx,
        )

        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )
        labels[labels > 1] = 1
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = -1

    def load_pokec(
        self,
        dataset,
        sens_attr,
        predict_attr,
        path="../dataset/pokec/",
        label_number=1000,
        sens_number=500,
        seed=19,
        test_idx=False,
    ):
        """Load data"""

        self.path_name = "pokec_n"
        self.url = "https://drive.google.com/u/0/uc?id=1wWm6hyCUjwnr0pWlC6OxZIj0H0ZSnGWs&export=download"
        self.destination = os.path.join(self.root, self.path_name, "pokec_n.zip")
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "region_job_2.csv")
        ):
            gdown.download(self.url, self.destination, quiet=False)
            with zipfile.ZipFile(self.destination, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "region_job_2_relationship.txt")
        ):
            gdown.download(self.url, self.destination, quiet=False)
            with zipfile.ZipFile(self.destination, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, self.path_name))

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "region_job_2.csv")
        )
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, "region_job_2_relationship.txt"),
            dtype=np.int64,
        )

        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random

        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)) :]

        sens = idx_features_labels[sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)

        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


class German(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "../dataset"):
        super(German, self).__init__(is_normalize=is_normalize, root=root)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_german("german")

        node_num = features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)

        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_german(
        self,
        dataset,
        sens_attr="Gender",
        predict_attr="GoodCustomer",
        path="../dataset/german/",
        label_number=100,
    ):
        # print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "german"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))

        if not os.path.exists(os.path.join(self.root, self.path_name, "german.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german.csv"
            file_name = "german.csv"
            self.download(url, file_name)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "german_edges.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german_edges.txt"
            file_name = "german_edges.txt"
            self.download(url, file_name)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("OtherLoansAtStore")
        header.remove("PurposeOfLoan")

        # Sensitive Attribute
        gender = idx_features_labels["Gender"].astype("string").str.strip()
        mapped_gender = gender.map({"Female": 1, "Male": 0})

        if mapped_gender.isna().any():
            bad = sorted(gender[mapped_gender.isna()].dropna().unique().tolist())
            raise ValueError(f"Unexpected Gender values in German dataset: {bad}")

        idx_features_labels["Gender"] = mapped_gender.astype("int64")
        
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].to_numpy(copy=True)
        labels[labels == -1] = 0
        
        labels = torch.LongTensor(labels)

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))

        labels = torch.LongTensor(labels)

        import random

        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0


class Credit(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "../dataset"):
        super(Credit, self).__init__(is_normalize=is_normalize, root=root)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_credit("credit")

        node_num = features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_credit(
        self,
        dataset,
        sens_attr="Age",
        predict_attr="NoDefaultNextMonth",
        path="../dataset/credit/",
        label_number=6000,
    ):
        from scipy.spatial import distance_matrix

        # print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "credit"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))

        if not os.path.exists(os.path.join(self.root, self.path_name, "credit.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/credit/credit.csv"
            file_name = "credit.csv"
            self.download(url, file_name)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "credit_edges.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/credit/credit_edges.txt"
            file_name = "credit_edges.txt"
            self.download(url, file_name)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("Single")

        # build relationship
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random

        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 1