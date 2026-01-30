<h1 align="center">Fairness in Link Prediction Beyond Demographic Parity: A Reproducibility Study</h1>
<h3 align="center">Valentijn Oldenburg, Floris de Kam, Stef de Wildt, Jarno Balk</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2511.06568">Original Paper</a> ·
  <a href="https://github.com/joaopedromattos/MORAL">Official MORAL implementation</a> ·
  <a href="/placeholder.pdf">Reproduced paper</a> 
</p>

This repository contains a reproducibility / extension project based on:

- **Paper:** *Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity* (arXiv:2511.06568)
- **Reference implementation:** **MORAL** (official repository linked above)

> This repository contains our reproducibility and extension study of MORAL (Mattos et al., 2025), a post-processing method for fair ranked link prediction. MORAL goes beyond dyadic demographic parity by focusing on exposure differences between pairs of subgroups. We rebuild the full experimental pipeline from the paper’s definitions, document and fix mismatches between the definitions and the released code, and provide a clear end-to-end workflow to run the experiments. Beyond reproduction, we test how stable MORAL’s fairness–utility trade-off is using other rank-aware fairness and utility metrics that are not designed around MORAL’s objective. We also extend the analysis to controlled synthetic graphs with different (a)symmetric homophily/heterophily settings and to sensitive attributes with more than two groups, giving a more detailed view of subgroup-pair disparities. Overall, the project supports exposure-based fairness as a useful way to study link prediction, while showing when ranking-based evaluation leads to reliable conclusions.

---


## Original Paper
- **Paper:** https://arxiv.org/abs/2511.06568  
- **Official MORAL code:** https://github.com/joaopedromattos/MORAL  

> automatic generation of MORAL-style splits (saved under `./data/splits/`).

---

## Repository layout

| Path | Description |
|------|-------------|
| `scripts/main.py` | Main training entry point (MORAL + multiclass trigger via dataset). |
| `scripts/models/moral.py` | MORAL model (per-sensitive-group encoders/decoders). |
| `scripts/models/moral_multiclass.py` | Multiclass MORAL variant (used for `credit_multiclass`). |
| `scripts/models/DPAH.py` | DPAH synthetic graph generator (homophily/heterophily control). |
| `scripts/helpers/utils.py` | Dataset loading + MORAL-style split generation + aggregation helpers. |
| `scripts/helpers/datasets.py` | Dataset definitions and download/preprocessing helpers. |
| `scripts/helpers/metrics.py` | Ranking/fairness metrics + results aggregation utilities. |
| `scripts/helpers/check_splits.py` | Validates saved split files and logs split statistics. |
| `scripts/helpers/create_splits.py` | Creates MORAL-style train/val/test edge splits (pos/neg, group-aware). |
| `scripts/fairadj.py` | FairAdj baseline runner (baseline code under `scripts/baseline/fairadj/`). |
| `scripts/fairwalk.py` | FairWalk baseline runner (baseline code under `scripts/baseline/fairwalk/`). |
| `scripts/detconstsort.py` | DetConstSort baseline runner (baseline code under `scripts/baseline/detconstsort`). |
| `scripts/homophily.py` | Synthetic DPAH / homophily experiments. |
| `scripts/results*.ipynb` | Notebooks used to aggregate results and create figures. |
| `data/splits/` | Saved MORAL-style edge splits (`<dataset>_<seed>.pt`). |
| `data/output/` | Model outputs (`three_classifiers_*.pt`) + final rankings + CSVs. |
| `dataset/` | Raw datasets (not included). |
| `emissions/` | CodeCarbon logs (emissions tracking). |
| `figures/` | Generated figures (produced by notebooks). |

---

# Setup

## Installing the packages


1. Create a Python environment (Python 3.9+ recommended).
2. Install PyTorch with CUDA support that matches your hardware by following the
   [official instructions](https://pytorch.org/).
3. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

# Notebooks

To recreate our results without rerunning the MORAL algorithms, use these notebooks...

All result notebooks live in [`./scripts/`](./scripts/):
- `results.ipynb` — main results aggregation
- `results_baselines.ipynb` — baseline comparisons
- `results_multiclass.ipynb` — multiclass results
- `results_xlayers.ipynb` — xlayers results
- `results_homophily.ipynb` — synthetic homophily experiments

---

## Datasets

Dataset classes and download helpers are in:
- [`./scripts/helpers/datasets.py`](./scripts/helpers/datasets.py)

### Datasets used in our experiments
This repo contains loaders for:
- `facebook`, `german`, `nba`, `credit`, `pokec_n`, `pokec_z`
- `airtraffic`, `chameleon` (homophily extension)
- `credit_multiclass` (multiclass extension)
- `dpah_*` (synthetic graphs; generated)

Raw dataset folders are stored under [`./dataset/`](./dataset/).

---

# Run MORAL

Entry point:
- [`./scripts/main.py`](./scripts/main.py)

Example (single dataset):
```bash
cd scripts

python main.py   --dataset facebook   --model gae   --fair_model moral   --device cuda:0   --epochs 300
```

Arguments (from `main.py`):
- `--dataset` (default: `facebook`)
- `--model` ∈ `{gae, ncn}`
- `--fair_model` (name only, used for logging/output filenames)
- `--device` (e.g. `cuda:0` or `cpu`)
- `--lr`, `--batch_size`, `--epochs`, `--hidden_dim`, `--weight_decay`
- `--runs` (default: 3), `--seed` (default: 0, increments by 1 per run)

---

## Run baselines

### FairAdj
Script:
- [`./scripts/fairadj.py`](./scripts/fairadj.py)

Run:
```bash
cd scripts
python fairadj.py --dataset facebook --device cuda:0
```

This line runs the fairadj.py implementation with default parameters, however if one wishes to change these, consider adding these additional following flags:
- `--outer_epochs` (default: 4)
- `--T1` (default: 50)
- `--T2` (default: 20)

> Outputs are saved to `./data/output/` for the evaluation notebooks.

### FairWalk
Script:
- [`./scripts/fairwalk.py`](./scripts/fairwalk.py)

Run:
```bash
cd scripts
python fairwalk.py --dataset facebook --device cuda:0
```

This line runs the fairwalk.py implementation with default parameters, however if one wishes to change these, consider adding these additional following flags:
- `--fair_model` (default: fairwalk)
- `--walks` (default: 20)
- `--walk_len` (default: 80)

### DetConstSort
Script:
- [`./scripts/detconstsort.py`](./scripts/detconstsort.py)

This method reranks outputs, so you typically run MORAL first:
```bash
cd scripts

# 1. (optionally) generate MORAL outputs 
python main.py --dataset facebook --model gae --fair_model moral --device cuda:0

# 2. rerank
python detconstsort.py --dataset facebook
```

---

## Multiclass extension

Multiclass MORAL is implemented in:
- [`./scripts/models/moral_multiclass.py`](./scripts/models/moral_multiclass.py)

In `main.py`, multiclass is only enabled when:
- `--dataset credit_multiclass`

Example:
```bash
cd scripts
python main.py --dataset credit_multiclass --model gae --fair_model moral_mc --device cuda:0 --epochs 300
```

---

## Synthetic homophily / DPAH

Synthetic graph generation + runs:
- [`./scripts/homophily.py`](./scripts/homophily.py)

Example:
```bash
cd scripts
python homophily.py --device cuda:0
```
This line runs the homophily.py implementation with default graph generation parameters, however if one wishes to change these, consider adding the following flags:
- `--fm_values` (default: `[0.05, 0.10, 0.20, 0.30, 0.40, 0.50]`)
- `--h_MM_values` (default: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`)
- `--h_mm_values"` (default: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`)
- `--N` (number of nodes)
- `--d` (desired edge density)
- `--plo_M` (Power-law outdegree parameter for majority class.)
- `--plo_m` (Power-law outdegree parameter for minority class.)
- `--feature_dim_DPAH` (Feature dimensionality used when wrapping DPAH as a dataset.)
- `--DPAH_seed` (Random seed used during DPAH graph generation.)

# Outputs

Outputs are saved under [`./data/output/`](./data/output/).

## Model outputs
`main.py` produces:
- `three_classifiers_<DATASET>_<FAIR_MODEL>_<MODEL>_<RUN>.pt`

Example:
- `three_classifiers_facebook_MORAL_GAE_0.pt`

## Final rankings
`main.py` also produces:
- `three_classifiers_<...>_final_ranking.pt`

Example:
- `three_classifiers_facebook_MORAL_GAE_0_final_ranking.pt`

---

# Acknowledgements

We thank the authors of **MORAL** for releasing both the paper and the reference implementation that this project builds on.  

We also acknowledge and credit the original authors and codebases used for the baseline methods included (or adapted) in this repository:

- **FairAdj**  
  Code: `scripts/baseline/fairadj/`  
  Paper: https://openreview.net/pdf?id=xgGS6PmzNq6  
  Official repository: https://github.com/brandeis-machine-learning/FairAdj  

- **FairWalk**  
  Code: `scripts/baseline/fairwalk/`  
  Paper: https://www.ijcai.org/proceedings/2019/0456.pdf  
  Official repository: https://github.com/urielsinger/fairwalk?tab=readme-ov-file  

- **DetConstSort / Fair Ranking (reranking baseline)**  
  Code: `scripts/baseline/detconstsort/`  
  Original paper introducing DetConstSort: https://arxiv.org/pdf/1905.01989  
  Follow-up paper referencing the original approach: https://dl.acm.org/doi/pdf/10.1145/3404835.3462850  
  Reference implementation used (from the follow-up work): https://github.com/evijit/SIGIR_FairRanking_UncertainInference/tree/main  

> If we missed or misattributed any baseline components, please contact us so we can correct the attribution.

---


# License
This project is released under the **MIT License**. See [`LICENSE`](./LICENSE)

---

# Contact
- Maintainer: F.P.J. de Kam
- E-mail: floris.de.kam@student.uva.nl
