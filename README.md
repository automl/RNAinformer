# RNAinformer

**RNAinformer: Generative RNA Design from Contact Maps**

This repository contains the source code for RNAinformer, a generative model for inverse
RNA folding from contact-map representations.

## Abstract

**Background:**
Inverse RNA folding designs sequences that realize a target RNA structure and is a core task in computational RNA engineering. Many design methods use dot-bracket encodings, which work well for nested base pair structures but do not naturally represent the full contact topologies produced by modern RNA structure predictors. Contact maps can encode pseudoknots, non-canonical contacts, and base multiplets in a single matrix representation. This leaves a representation gap: RNA structure prediction often produces contact maps, while RNA design methods largely remain tied to string based structure inputs.

**Results:**
We present RNAinformer, a generative model for RNA sequence design from contact maps. RNAinformer uses an encoder-decoder transformer with axial-attention to process binary nucleotide interaction matrices and supports positional sequence constraints and global GC content conditioning. To reduce homology driven overestimation, RNAinformer is trained on synthetic Rfam derived data with family and clan separation. We evaluate the model from canonical nested inverse folding to pseudoknotted and experimentally derived contact map design tasks. In the nested setting, RNAinformer outperforms learning based baselines under a fixed sampling budget, while the specialized search method SAMFEO achieves the highest solved rate. On pseudoknotted targets, RNAinformer solves 39.1% of the tasks with a single candidate design compared with 15.6% for antaRNA, an established pseudoknot capable baseline. For experimentally derived contact maps containing heterogeneous interactions, RNAinformer generates valid designs for a subset of targets and achieves high best candidate binary contact map agreement. Designed sequences also show consistent agreement across independent secondary structure predictors.

**Conclusions:**
RNAinformer extends RNA design from string encodings to contact map representations. The same model interface handles nested structures, pseudoknots, constrained sequence design, and contact maps containing non-canonical contacts and degree-greater-than-one topologies. These results support contact maps as a useful representation for connecting RNA structure prediction and sequence design.

### Install virtual environment

```
conda env create -f environment.yml
conda activate rnadesign
```
The Flash Attention package currently requires an Ampere, Ada, or Hopper GPU (e.g., A100,
RTX 3090, RTX 4090, H100). To install Flash-Attention:
```
pip install -U --no-cache-dir flash-attn==2.3.4
```

### Datasets
Download and unzip the training and test sets. Archived (citable) copy on Zenodo:
**DOI: [10.5281/zenodo.21746333](https://doi.org/10.5281/zenodo.21746333)**. Convenience mirror (Dropbox):
```
wget -O data.tar.xz "https://www.dropbox.com/scl/fi/yaxvlsloht21i7bho2tim/data.tar.xz?rlkey=jmxqbjjcmbumt08hk2tbqxvgg&st=k9jfe7iz&dl=1"
tar -xvf data.tar.xz
rm data.tar.xz
```

### Models and predictions
Download and unzip the trained models and generated predictions. Archived (citable) copy on
Zenodo: **DOI: [10.5281/zenodo.21746586](https://doi.org/10.5281/zenodo.21746586)**. Convenience mirror (Dropbox):
```
wget -O runs.tar.xz "https://www.dropbox.com/scl/fi/4ti5cn1zuct5u37rzkpod/runs.tar.xz?rlkey=jfu6trrvnr9d118mrsecgquzp&st=eccnnqy8&dl=1"
tar -xvf runs.tar.xz
rm runs.tar.xz
```

### Reproducing the manuscript results
Precomputed metrics are provided in the respective `metrics.csv` files. To recompute all
evaluations from the predictions, there are two equivalent routes.

**Route A — repository/Dropbox archive (native `runs/` layout).**
After extracting `data.tar.xz` and `runs.tar.xz` as above (so `./data` and `./runs` exist):
```
bash run_evaluations.sh
```

**Route B — Zenodo record (human-readable browse layout).**
The Zenodo `predictions` archive stores predictions in a human-readable browse layout
(`predictions/rnainformer/<model>/version_*/...` plus `predictions/baselines/...`).
`adapt_to_eval_layout.sh` maps that layout onto the `runs/` layout expected by
`run_evaluations.sh` using symlinks only (no copies):
```
# after downloading + extracting the Zenodo predictions/ and datasets/ folders:
bash adapt_to_eval_layout.sh predictions runs
cp -r datasets data          # or: ln -s datasets data
bash run_evaluations.sh
```
Both routes produce the same reported metrics.

### Inference on test sets
```
python inference.py --seed 9647359 --path path/to/model/folder/
```
E.g.:
```
python inference.py --seed 9647359 --path runs/syn_pdb/version_0/
```
Use `--flash False` if Flash Attention is not installed:
```
python inference.py --seed 9647359 --path runs/syn_pdb/version_0/ --flash False
```

### Availability
- Source code: this repository, Apache-2.0 license. Archived version for the manuscript:
  Zenodo **DOI: [10.5281/zenodo.21747282](https://doi.org/10.5281/zenodo.21747282)**.
- Data, trained models, generated candidates, prediction outputs, configs, and metric
  files: Zenodo **DOIs: datasets [10.5281/zenodo.21746333](https://doi.org/10.5281/zenodo.21746333); trained models, candidates, and predictions [10.5281/zenodo.21746586](https://doi.org/10.5281/zenodo.21746586)**.

### Contribution
This repository is a copy of the original source code for reasons of maintenance.
The original source code is available at <https://github.com/pilar12/RNA-design>.
