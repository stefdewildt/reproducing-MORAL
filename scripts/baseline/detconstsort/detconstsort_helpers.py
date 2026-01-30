# Author: Gosh et al., modified by F.P.J. de Kam (floris.de.kam@student.uva.nl)

from __future__ import annotations
import math
import torch


@torch.no_grad()
def detconstsort_rerank(
    outputs: torch.Tensor,
    test_labels: torch.Tensor,
    edge_sens_groups: torch.Tensor,
    pi: torch.Tensor,
    K: int = 1000,
    kmax: int | None = None
):
    """
    Reranking method according to DETCONSTSORT algorithm from 
    """

                                        # set kmax if not provided
    if kmax is None:
        kmax = int(K) - 1  

    outputs = outputs.view(-1).detach().cpu()
    test_labels = test_labels.view(-1).detach().cpu()
    edge_sens_groups = edge_sens_groups.view(-1).long().detach().cpu()
    pi = pi.view(-1).detach().cpu()
                                        # validate input lengths
    N = int(outputs.numel())
    if N != int(test_labels.numel()) or N != int(edge_sens_groups.numel()):
        raise ValueError("outputs, test_labels, edge_sens_groups must have same length")

    G = int(pi.numel())

                                        # global ranking order, basis for reranking
    global_sorted = torch.argsort(outputs, descending=True)

                                        # scorelist analogous to repo
    scoreList = []
    for r, idx in enumerate(global_sorted.tolist(), start=1):
        g = int(edge_sens_groups[idx].item())
        s = float(outputs[idx].item())
        y = float(test_labels[idx].item())
                                        # append idx for remapping later
        scoreList.append((g, s, y, r, r, idx))  

                                        # unique attribute groups
    AttrList = list(set([e[0] for e in scoreList]))

    AttrScores = {}
    AttrCount = {}
    minAttrCount = {}
    GlobalAttrCounts = {}

    for attr in AttrList:
        AttrCount[attr] = 0
        minAttrCount[attr] = 0
        GlobalAttrCounts[attr] = sum(1 for e in scoreList if e[0] == attr)
                                        # keep global scores per attribute for picking later
        AttrScores[attr] = [(e[1], e[5]) for e in scoreList if e[0] == attr]  

                                        # get pi for each attribute
    prob = {g: float(pi[g].item()) for g in range(G)}
                                        # assure all attributes have a probability entry
    for attr in AttrList:
        if attr not in prob:
            prob[attr] = 0.0

    rankedAttrList = []
    rankedScoreList = []                # list of (score, idx)
    maxIndices = []

    lastEmpty = 0
    k = 0

    while lastEmpty <= kmax:
        if lastEmpty == len(scoreList):
            break

        k += 1
        tempMinAttrCount = {}
        changedMins = {}
                                        # determine new min counts per attribute
        for attr in AttrList:
            tempMinAttrCount[attr] = math.floor(k * prob[attr])
            if (
                minAttrCount[attr] < tempMinAttrCount[attr]
                and minAttrCount[attr] < GlobalAttrCounts[attr]
            ):
                changedMins[attr] = AttrScores[attr][AttrCount[attr]]  

        if len(changedMins) != 0:
                                        # sort changedMins on score descending
            ordChangedMins = sorted(changedMins.items(), key=lambda x: x[1][0], reverse=True)

            for attr, (score, idx) in ordChangedMins:
                rankedAttrList.append(attr)
                rankedScoreList.append((score, idx))
                maxIndices.append(k)
                start = lastEmpty
                                        # bubble up to maintain score order
                while (
                    start > 0
                    and maxIndices[start - 1] >= start
                    and rankedScoreList[start - 1][0] < rankedScoreList[start][0]
                ):
                                        # swap in all 3 lists
                    rankedScoreList[start - 1], rankedScoreList[start] = rankedScoreList[start], rankedScoreList[start - 1]
                    maxIndices[start - 1], maxIndices[start] = maxIndices[start], maxIndices[start - 1]
                    rankedAttrList[start - 1], rankedAttrList[start] = rankedAttrList[start], rankedAttrList[start - 1]
                    start -= 1

                AttrCount[attr] += 1
                lastEmpty += 1

                                        # update minAttrCount only for changed attributes
            minAttrCount = dict(tempMinAttrCount)

                                        # output tensors
    picked_idx = torch.tensor([idx for (_, idx) in rankedScoreList], dtype=torch.long)
    final_scores = outputs[picked_idx]
    final_labels = test_labels[picked_idx]
    final_groups = edge_sens_groups[picked_idx]
    
    return final_scores, final_labels, final_groups, picked_idx
