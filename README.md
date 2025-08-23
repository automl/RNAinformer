# RNAinformer

This repository contains the source code to RNAinformer: Generative RNA Design with Tertiary Interactions.

The function of an RNA molecule depends on its structure and a strong structure-to-function relationship is already achieved on the secondary structure level of RNA.
Therefore, a common approach to RNA design is secondary structure based inverse RNA folding, where the goal of the design algorithm is to invert a RNA secondary structure prediction algorithm.
However, existing RNA design methods cannot invert all folding algorithms because they cannot represent all types of base interactions.
In this work, we propose RNAinformer, a novel transformer based approach to the inverse RNA folding problem.
Leveraging axial-attention, we directly model the secondary structure input represented as an adjacency matrix in a 2D latent space, which allows us to invert all existing secondary structure prediction algorithms.
Consequently, RNAinformer is the first model capable of designing RNAs from secondary structures with all kinds of nucleotide interactions, including non-canonical base pairs and tertiary interactions like pseudoknots and base multiplets.
Our comprehensive evaluations at different levels of secondary structure complexity, across multiple RNA design tasks, including conditional generation with desired GC-contents and involving multiple RNA folding algorithms, reveal RNAinformer's outstanding RNA design capabilities.

### Install virtual environment


```
conda env create -f environment.yml

conda activate rnadesign

```
The Flash Attention package currently requires an Ampere, Ada, or Hopper GPU (e.g., A100, RTX 3090, RTX 4090, H100). To install Falsh-attn.

```
pip install -U --no-cache-dir flash-attn==2.3.4
```
### Datasets
To get the training and test sets download and unzip data from https://www.dropbox.com/scl/fi/yaxvlsloht21i7bho2tim/data.tar.xz?rlkey=jmxqbjjcmbumt08hk2tbqxvgg&st=k9jfe7iz&dl=0

```
wget -O data.tar.xz https://www.dropbox.com/scl/fi/yaxvlsloht21i7bho2tim/data.tar.xz?rlkey=j
mxqbjjcmbumt08hk2tbqxvgg&st=k9jfe7iz&dl=0
tar -xvf data.tar.xz
rm data.tar.xz
```

### Models and predictions
To get the models and designed predictions download and unzip data from https://www.dropbox.com/scl/fi/4ti5cn1zuct5u37rzkpod/runs.tar.xz?rlkey=jfu6trrvnr9d118mrsecgquzp&st=eccnnqy8&dl=0

```
wget -O runs.tar.xz https://www.dropbox.com/scl/fi/4ti5cn1zuct5u37rzkpod/runs.tar.xz?rlkey=jfu6tr
rvnr9d118mrsecgquzp&st=eccnnqy8&dl=0
tar -xvf runs.tar.xz
rm runs.tar.xz
```

### Evaluate 
Evaluation metrics have been provided for all model and competitors in respective metrics.csv files
To run all evaluations again

```
bash run_evaluation.sh
```

### Inference on test sets
```
python inference.py --seed 9647359 --path path/to/model/folder/
```
Eg.
```
python inference.py --seed 9647359 --path runs/syn_pdb/version_0/
```
Use --flash False if Flash attention is not installed
```
python inference.py --seed 9647359 --path runs/syn_pdb/version_0/ --flash False
```

### Contribution
This repository is a copy of the original source code for reasons of maintenance.
The original source code is available at <https://github.com/pilar12/RNA-design>.
