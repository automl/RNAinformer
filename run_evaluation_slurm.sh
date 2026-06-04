#!/bin/bash
# =============================================================================
# RNAinformer - SLURM job: validate the (always-flash) GPU setup, then run the
# full evaluation (run_evaluations.sh) over all models / competitors.
#
# Submit with:   sbatch run_evaluation_slurm.sh
# Validate only: VALIDATE_ONLY=1 sbatch run_evaluation_slurm.sh
#
# The evaluation itself (eval.py -> ViennaRNA folding + metrics) is CPU-bound and
# reads the pre-computed predictions in runs/**/predictions*/. A GPU is requested
# only for the preflight step, which exercises the real always-flash forward pass
# so you know the model runs on this cluster before the long CPU eval starts.
# If you do NOT need the GPU validation, drop "--gres" below and set VALIDATE_ONLY=0
# together with SKIP_GPU_CHECK=1.
# =============================================================================

#SBATCH --job-name=rnainformer-eval
#SBATCH --partition=gpu                 # <-- set to your cluster's GPU partition
##SBATCH --account=your_account         # <-- uncomment + set if your cluster needs it
#SBATCH --gres=gpu:1                     # one GPU (for the always-flash preflight)
#SBATCH --cpus-per-task=8                # eval folds sequences on CPU (ViennaRNA)
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# ----------------------------- configuration ---------------------------------
ENV_NAME="${ENV_NAME:-rnadesign}"                       # conda env name
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"     # repo root
VALIDATE_MODEL_DIR="${VALIDATE_MODEL_DIR:-runs/syn_pdb/version_0}"  # model used for preflight
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"                     # 1 = stop after preflight
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"                   # 1 = skip GPU/flash validation
# -----------------------------------------------------------------------------

cd "$REPO_DIR"
mkdir -p slurm_logs

echo "================ RNAinformer cluster run ================"
echo "host        : $(hostname)"
echo "repo        : $REPO_DIR"
echo "conda env   : $ENV_NAME"
echo "time        : $(date)"
echo "========================================================="

# --- activate conda (works in non-interactive SLURM shells) ---
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/.conda")}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate "$ENV_NAME"
echo "python: $(which python)"

# ----------------------------- preflight -------------------------------------
if [ "$SKIP_GPU_CHECK" != "1" ]; then
    echo "=== GPU ==="
    nvidia-smi || { echo "ERROR: nvidia-smi failed - no GPU on this node"; exit 1; }

    echo "=== Preflight: validate the always-flash forward path on GPU ==="
    python - "$VALIDATE_MODEL_DIR" <<'PY'
import sys, os, yaml, torch
from glob import glob
path = sys.argv[1]

# 1) CUDA available
assert torch.cuda.is_available(), "CUDA not available on this node"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0))

# 2) torch >= 2.1 -- required for F.scaled_dot_product_attention(..., scale=...)
#    NOTE: environment.yml pins torch==2.0.1, which does NOT accept the `scale`
#    kwarg and will crash the sequence encoder. Upgrade torch on the cluster.
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert (major, minor) >= (2, 1), (
    f"torch>=2.1 required for the SDPA `scale=` kwarg, found {torch.__version__}. "
    "Upgrade torch (environment.yml's 2.0.1 is too old).")
print("torch version OK (>= 2.1)")

# 3) flash-attn importable (always-flash architecture needs it on GPU)
import flash_attn
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func  # noqa: F401
print("flash-attn", getattr(flash_attn, "__version__", "?"), "OK")

# 4) ViennaRNA -- required by the evaluation
from RNA import fold  # noqa: F401
print("ViennaRNA OK")

# 5) build always-flash model + STRICT-load the released checkpoint
sys.path.insert(0, os.getcwd())
from RNAinformer.utils.configuration import Config
from RNAinformer.model.RNADesignFormer import RNADesignFormer
from RNAinformer.pl_modules.rna_datamodule import IGNORE_INDEX, PAD_INDEX
from RNAinformer.utils.data.rna import CollatorRNADesignMat

cfgd = yaml.load(open(os.path.join(path, "config.yaml")), Loader=yaml.Loader)
assert cfgd["RNADesignFormer"].get("flash") is True, "config flash must be True (always-flash)"
ckpt = glob(os.path.join(path, "checkpoints", "*.ckpt"))[0]
cfgd["model_path"] = ckpt
cfg = Config(config_dict=cfgd)
model = RNADesignFormer(cfg.RNADesignFormer)
raw = torch.load(ckpt, map_location="cpu")
sd = {k.replace("model.", ""): v for k, v in raw.items() if "model." in k}
model.load_state_dict(sd, strict=True)
print("strict-load OK:", os.path.basename(ckpt))

# 6) real always-flash forward on GPU (fp16, as inference.py does) -- one batch
model.cuda().eval().to(torch.float16)
collator = CollatorRNADesignMat(PAD_INDEX, IGNORE_INDEX)
ds = cfg.test.datasets[0]
data = [x for x in torch.load(f"{cfg.test.cache_dir}/{ds}.pt") if x["length"] <= cfg.rna_data.max_len][:4]
batch = collator(data)
with torch.no_grad():
    out = model.generate(batch["src_struct"].cuda(), batch["length"].cuda(), None, None, None, greedy=True)
assert out.shape[0] == len(data)
print("flash forward OK, output shape:", tuple(out.shape))
print("PREFLIGHT PASSED")
PY
else
    echo "=== Skipping GPU/flash preflight (SKIP_GPU_CHECK=1) ==="
fi

if [ "$VALIDATE_ONLY" = "1" ]; then
    echo "VALIDATE_ONLY=1 -> stopping after preflight."
    exit 0
fi

# ----------------------------- full evaluation -------------------------------
echo "=== Full evaluation: bash run_evaluations.sh ==="
echo "(reads runs/**/predictions*/, folds sequences via ViennaRNA, writes metrics.csv)"
bash run_evaluations.sh

echo "=== Done. Per-model metrics.csv under runs/**/ ; combined output from comb_metrics.py ==="
echo "finished: $(date)"
