# Author: Mattos et al., modified by F.P.J. de Kam (floris.de.kam@student.uva.nl)

from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .moral import EdgeLabelDataset, build_encoder, build_predictor, normalize_features
from helpers.create_splits import edge_group_ids


################################################################
# MultiClass adaptation of MORAL
################################################################

class MORALMultiClass(nn.Module):
    def __init__(
        self,
        adj: Tensor,
        features: Tensor,
        labels: Tensor,
        idx_train: Tensor,
        idx_val: Tensor,
        idx_test: Tensor,
        sens: Tensor,
        sens_idx: Tensor,
        edge_splits: Dict[str, Dict[str, Tensor]],
        dataset_name: str,
        num_hidden: int = 128,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        encoder: str = "gcn",
        encoder_layers: int = 2,
        decoder: str = "gae",
        batch_size: int = 1024,
        device: str = "cpu",
        num_sens_classes: int = 3,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.criterion = nn.BCEWithLogitsLoss()

        self.encoder_layers = int(encoder_layers)
        if self.encoder_layers < 1:
            raise ValueError("encoder_layers must be >= 1")

        self.features = normalize_features(features.float()).to(self.device)
        self.edge_index = adj.coalesce().indices().to(self.device)
        self.labels = labels.to(self.device)

                                        # sens must be 1D LongTensor
        self.sens = sens.view(-1).long().to(self.device)
        self.sens_cpu = self.sens.cpu()

        self.sens_idx = sens_idx
                                        # these indices are not used in the current MORAL training loop (edge-split based),
                                        # but we keep them for parity with moral.py / existing code.
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        self.num_classes = int(num_sens_classes)
        if self.num_classes < 2:
            raise ValueError("num_sens_classes must be >= 2")

                                        # unordered dyads => C*(C+1)/2 groups
        self.num_groups = self.num_classes * (self.num_classes + 1) // 2

                                        # sanity check on sens range
        smin = int(self.sens.min().item()) if self.sens.numel() else 0
        smax = int(self.sens.max().item()) if self.sens.numel() else -1
        if self.sens.numel() and (smin < 0 or smax >= self.num_classes):
            raise ValueError(
                f"sens contains values outside [0, {self.num_classes - 1}]: min={smin}, max={smax}"
            )
                                        # build encoders and predictors
        self.encoders = nn.ModuleList([
            build_encoder(encoder, self.features.size(1), num_hidden, layers=self.encoder_layers).to(self.device)
            for _ in range(self.num_groups)
        ])
        self.predictors = nn.ModuleList([
            build_predictor(decoder, num_hidden).to(self.device)
            for _ in range(self.num_groups)
        ])
                                        # make optimiser for each group
        self.optimizers = [
            torch.optim.Adam(
                list(self.encoders[g].parameters()) + list(self.predictors[g].parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            for g in range(self.num_groups)
        ]

        self.train_loaders = self._build_group_loaders(edge_splits.get("train"), shuffle=True)
        self.valid_loaders = self._build_group_loaders(edge_splits.get("valid"), shuffle=False)

        test_edges, test_labels = self._prepare_edges(edge_splits.get("test"))
        self.test_edges = test_edges
        self.test_labels = test_labels

        self.original_sens_dist = self._compute_original_sens_dist()
        self.best_state: Optional[Dict[str, Tensor]] = None

        logger.info(
            "MORALMultiClass init | dataset={} | encoder={} | layers={} | num_classes={} | num_groups={}",
            self.dataset_name,
            encoder,
            self.encoder_layers,
            self.num_classes,
            self.num_groups,
        )

    ################################################################
    # Original unchanged functions
    ################################################################
    def _prepare_edges(self, split: Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor]:
        if split is None:
            return torch.empty(0, 2, dtype=torch.long), torch.empty(0)

        pos_edges = split["edge"].long()
        neg_edges = split["edge_neg"].long()
        edges = torch.cat([pos_edges, neg_edges], dim=0)
        labels = torch.cat(
            [torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))], dim=0
        )
        return edges, labels

    def _build_group_loaders(self, split: Optional[Dict[str, Tensor]], shuffle: bool) -> List[Optional[DataLoader]]:
        if split is None:
            return [None] * self.num_groups

        edges, labels = self._prepare_edges(split)
        sens_groups = edge_group_ids(edges, self.sens_cpu, self.num_classes)

        loaders: List[Optional[DataLoader]] = []
        for group in range(self.num_groups):
            mask = sens_groups == group
            if mask.sum() == 0:
                loaders.append(None)
                continue

            dataset = EdgeLabelDataset(edges[mask], labels[mask])
            batch = len(dataset) if self.batch_size <= 0 else self.batch_size
            loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=False)
            loaders.append(loader)

        return loaders

    def _compute_original_sens_dist(self) -> Tensor:
        src, dst = self.edge_index
        edges = torch.stack([src, dst], dim=1).to(self.sens_cpu.device)
        edge_groups = edge_group_ids(edges, self.sens_cpu, self.num_classes)
        counts = torch.bincount(edge_groups, minlength=self.num_groups).float()
        return counts / counts.sum().clamp(min=1.0)

    def forward(self, group: int) -> Tensor:
        return self.encoders[group](self.features, self.edge_index)

    def _train_epoch(self) -> float:
        total_loss = 0.0
        total_batches = 0

        for group, loader in enumerate(self.train_loaders):
            if loader is None:
                continue

            encoder = self.encoders[group]
            predictor = self.predictors[group]
            optimizer = self.optimizers[group]

            encoder.train()
            predictor.train()

            for edges, labels in loader:
                optimizer.zero_grad()
                edges = edges.t().to(self.device)
                labels = labels.to(self.device)

                embeddings = encoder(self.features, self.edge_index)
                logits = predictor(embeddings, edges)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loaders: Sequence[Optional[DataLoader]]) -> Optional[float]:
        total_loss = 0.0
        total_batches = 0

        for group, loader in enumerate(loaders):
            if loader is None:
                continue

            self.encoders[group].eval()
            self.predictors[group].eval()

            embeddings = self.encoders[group](self.features, self.edge_index)
            for edges, labels in loader:
                edges = edges.t().to(self.device)
                labels = labels.to(self.device)
                logits = self.predictors[group](embeddings, edges)
                loss = self.criterion(logits, labels)
                total_loss += float(loss.item())
                total_batches += 1

        if total_batches == 0:
            return None
        return total_loss / total_batches

    def fit(self, epochs: int = 300) -> None:
        best_val = float("inf")
        best_state: Optional[Dict[str, Tensor]] = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._evaluate(self.valid_loaders)

            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(self.state_dict())

            if epoch % 10 == 0 or epoch == 1:
                if val_loss is None:
                    logger.info("Epoch {} | train={:.4f}", epoch, train_loss)
                else:
                    logger.info("Epoch {} | train={:.4f} | valid={:.4f}", epoch, train_loss, val_loss)

        if best_state is not None:
            self.load_state_dict(best_state)
            self.best_state = best_state

    @torch.no_grad()
    def predict(self) -> Tensor:
        if self.test_edges.numel() == 0:
            raise RuntimeError("Test split is empty. Did you load the dataset correctly?")

        outputs = torch.zeros(self.test_edges.size(0), device=self.device)
        sens_groups = edge_group_ids(self.test_edges, self.sens_cpu, self.num_classes)

        for group in range(self.num_groups):
            mask = sens_groups == group
            if mask.sum() == 0:
                continue

            edges = self.test_edges[mask].t().to(self.device)
            self.encoders[group].eval()
            self.predictors[group].eval()
            embeddings = self.encoders[group](self.features, self.edge_index)
            logits = self.predictors[group](embeddings, edges)
            outputs[mask] = logits

        return outputs