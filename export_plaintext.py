#!/usr/bin/env python
"""
Export all designed RNA sequences (and their design targets) from the
pre-computed predictions in ``runs/`` readable.

For every (method, dataset) combination we write a subfolder
``export_plaintext/<method>/<dataset>/`` containing:

  * ``targets.csv``  - one row per design task (the *target*: structure in
                       dot-bracket notation, base pairs, reference sequence,
                       GC target, pseudoknot/multiplet flags ...).
  * ``designs.csv``  - one row per designed candidate sequence (plain ACGU),
                       self-contained: the target structure (and, for the
                       riboswitch partial-design task, the partial target
                       sequence) is repeated on every row.
  * ``data.json``    - the same information nested: each target with the list
                       of sequences designed for it (target + designs together).

Numeric encodings used inside ``runs/``:
  sequence   1=A 2=C 3=G 4=U 5=N      (0=BOS only in stored reference seqs)
  structure  0=. 1=( 2=) 3=[ 4=] 5={ 6=} 7=< 8=>   (matched bracket pairs =
             base pairs; different bracket types encode (pseudo)knots)

Run from the repo root (needs the downloaded ``runs/`` + ``data/`` bundles):
    python export_plaintext.py
"""

import os
import re
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

OUT_ROOT = "export_plaintext"

SEQ_ITOS = {0: "", 1: "A", 2: "C", 3: "G", 4: "U", 5: "N"}
STRUCT_ITOS = {0: ".", 1: "(", 2: ")", 3: "[", 4: "]", 5: "{", 6: "}", 7: "<", 8: ">"}


def dec_seq(ints):
    return "".join(SEQ_ITOS.get(int(i), "?") for i in list(ints))


def dec_struct(ints):
    return "".join(STRUCT_ITOS.get(int(i), "?") for i in list(ints))


def base_pairs(x):
    """Target base pairs (0-indexed) from pos1id/pos2id; loss-less for pdb."""
    p1 = x["pos1id"].tolist()
    p2 = x["pos2id"].tolist()
    return [(int(a), int(b)) for a, b in zip(p1, p2)]


def bp_str(pairs):
    return ";".join(f"{a}-{b}" for a, b in pairs)


def seed_of(path):
    m = re.search(r"version_(\d+)", path)
    return int(m.group(1)) if m else 0


def load_test(pt, max_len=200, filter_len=True):
    data = torch.load(pt, weights_only=False)
    if filter_len:
        data = [x for x in data if int(x["length"]) <= max_len]
    return data


def target_record_syn(tid, x, gc_controlled):
    """Common target record for the synthetic / pdb / antaRNA design tasks."""
    return {
        "target_id": tid,
        "length": int(x["length"]),
        "target_structure": dec_struct(x["src_struct"].tolist()),
        "target_base_pairs": bp_str(base_pairs(x)),
        "reference_sequence": dec_seq(x["trg_seq"].tolist()),
        "gc_content": round(float(x["gc_content"]), 6),
        "gc_controlled": bool(gc_controlled),
        "has_pseudoknot": bool(int(x["has_pk"])),
        "has_multiplet": bool(int(x["has_multiplet"])),
        "has_non_canonical": bool(int(x["has_nc"])),
    }


def write_folder(method, dataset, targets, designs, extra_note=""):
    """targets: dict target_id -> target record. designs: list of design rows."""
    folder = os.path.join(OUT_ROOT, method, dataset)
    os.makedirs(folder, exist_ok=True)

    tdf = pd.DataFrame(list(targets.values())).sort_values("target_id")
    tdf.to_csv(os.path.join(folder, "targets.csv"), index=False)

    ddf = pd.DataFrame(designs)
    # stable, readable ordering
    sort_cols = [c for c in ["target_id", "seed", "candidate_id"] if c in ddf.columns]
    ddf = ddf.sort_values(sort_cols)
    ddf.to_csv(os.path.join(folder, "designs.csv"), index=False)

    # nested JSON: each target with its designs
    by_target = defaultdict(list)
    design_keys = [c for c in ddf.columns if c not in ("target_id",)]
    for row in ddf.to_dict("records"):
        by_target[row["target_id"]].append({k: row[k] for k in design_keys})
    nested = {
        "method": method,
        "dataset": dataset,
        "note": extra_note,
        "n_targets": int(len(targets)),
        "n_designs": int(len(designs)),
        "targets": [
            {**targets[tid], "designs": by_target[tid]}
            for tid in sorted(targets.keys())
        ],
    }
    with open(os.path.join(folder, "data.json"), "w") as f:
        json.dump(nested, f, indent=1, default=str)

    print(f"  -> {method}/{dataset}: {len(targets)} targets, {len(designs)} designs")
    return {
        "method": method,
        "dataset": dataset,
        "n_targets": len(targets),
        "n_designs": len(designs),
    }


# ---------------------------------------------------------------------------
# Generic synthetic / pdb exporter (RNAinformer + LEARNA-suite + SAMFEO)
# ---------------------------------------------------------------------------
def export_syn(
    method, dataset, pred_paths, test_pt, kind, gc_controlled, filter_len=True
):
    """
    kind: 'seq'        -> sequence only            (syn_ns, learna, samfeo)
          'seq_db'     -> sequence + dot-bracket   (syn_hk)
          'seq_bp'     -> sequence + base-pair list(pdb)
    pred_paths: list of (seed, path)
    """
    test = load_test(test_pt, filter_len=filter_len)
    targets = {
        i: target_record_syn(i, test[i], gc_controlled) for i in range(len(test))
    }

    designs = []
    for seed, path in pred_paths:
        df = pd.read_pickle(path, compression="tar")
        for tid, grp in df.groupby("id"):
            tid = int(tid)
            if tid not in targets:
                continue
            L = targets[tid]["length"]
            for _, r in grp.sort_values("p_id").iterrows():
                seq = dec_seq(r["sequence"])
                assert (
                    len(seq) == L
                ), f"{method}/{dataset} id={tid}: len {len(seq)}!={L}"
                row = {
                    "target_id": tid,
                    "seed": seed,
                    "candidate_id": int(r["p_id"]),
                    "designed_sequence": seq,
                    "length": L,
                    "target_structure": targets[tid]["target_structure"],
                    "gc_content": targets[tid]["gc_content"],
                }
                if kind == "seq_db":
                    row["predicted_structure"] = r["structure"]
                elif kind == "seq_bp":
                    row["predicted_base_pairs"] = bp_str(
                        [(int(a), int(b)) for a, b in r["structure"]]
                    )
                designs.append(row)
    return write_folder(method, dataset, targets, designs)


# ---------------------------------------------------------------------------
# antaRNA: predictions are [target_db, designed_seq, folded_db] triples (plain
# text already); one candidate per target.
# ---------------------------------------------------------------------------
def export_antarna(method, pred_path, gc_controlled):
    test = load_test("data/syn_hk/syn_hk_test_antarna.pt")
    preds = pd.read_pickle(pred_path, compression="tar")
    if len(preds) != len(test):  # known off-by-one (drop idx 8)
        preds = [preds[i] for i in range(len(preds)) if i != 8]
    assert len(preds) == len(test), (len(preds), len(test))

    targets = {
        i: target_record_syn(i, test[i], gc_controlled) for i in range(len(test))
    }
    designs = []
    for i, (tgt_db, seq, folded_db) in enumerate(preds):
        assert len(tgt_db) == targets[i]["length"]
        designs.append(
            {
                "target_id": i,
                "seed": 0,
                "candidate_id": 0,
                "designed_sequence": seq,
                "length": targets[i]["length"],
                "target_structure": targets[i]["target_structure"],
                "gc_content": targets[i]["gc_content"],
                "predicted_structure": folded_db,
            }
        )
    return write_folder(
        method,
        "syn_hk",
        targets,
        designs,
        extra_note="antaRNA produces one candidate per target.",
    )


# ---------------------------------------------------------------------------
# Riboswitch partial design (RNAinformer ribo / ribo_gc)
# ---------------------------------------------------------------------------
def export_ribo(method, pred_paths, gc):
    spec = pd.read_pickle(
        "data/riboswitch/ribo_design_all.plk"
    )  # 1440 partial-design tasks
    gc_bands = (
        torch.load("data/riboswitch/gc_bands_ribo.pt", weights_only=False)
        if gc
        else None
    )

    def ribo_target(tid, ribo_idx, gc_band=None):
        row = spec.iloc[ribo_idx]
        rec = {
            "target_id": tid,
            "ribo_target_index": int(ribo_idx),
            "length": len(row["target_sequence"]),
            "target_structure": row["target_structure"],  # partial, with N
            "target_sequence": row[
                "target_sequence"
            ],  # partial: fixed aptamer + N to design
            "gc_controlled": bool(gc),
        }
        if gc_band is not None:
            rec["target_gc_band"] = float(gc_band)
        return rec

    targets = {}
    designs = []

    if not gc:
        for ribo_idx in range(len(spec)):
            targets[ribo_idx] = ribo_target(ribo_idx, ribo_idx)
        for seed, path in pred_paths:
            df = pd.read_pickle(path, compression="tar")
            for tid, grp in df.groupby("id"):
                tid = int(tid)
                tlen = targets[tid]["length"]
                for _, r in grp.sort_values("p_id").iterrows():
                    seq = dec_seq(r["sequence"])
                    assert len(seq) == tlen, f"ribo id={tid}: {len(seq)}!={tlen}"
                    designs.append(
                        {
                            "target_id": tid,
                            "ribo_target_index": tid,
                            "seed": seed,
                            "candidate_id": int(r["p_id"]),
                            "designed_sequence": seq,
                            "length": tlen,
                            "target_structure": targets[tid]["target_structure"],
                            "target_sequence": targets[tid]["target_sequence"],
                        }
                    )
    else:
        # composite target id over (gc_band, ribo_target_index)
        key2tid = {}

        def get_tid(gc_band, ribo_idx):
            key = (round(float(gc_band), 4), int(ribo_idx))
            if key not in key2tid:
                tid = len(key2tid)
                key2tid[key] = tid
                targets[tid] = ribo_target(tid, ribo_idx, gc_band)
            return key2tid[key]

        for seed, path in pred_paths:
            df = pd.read_pickle(path, compression="tar")
            for gc_band, gdf in df.groupby("gc"):
                idx_map = gc_bands[float(gc_band)]  # band id -> original ribo index
                for band_id, grp in gdf.groupby("id"):
                    ribo_idx = int(idx_map[int(band_id)])
                    tid = get_tid(gc_band, ribo_idx)
                    tlen = targets[tid]["length"]
                    for _, r in grp.sort_values("p_id").iterrows():
                        seq = dec_seq(r["sequence"])
                        assert (
                            len(seq) == tlen
                        ), f"ribo_gc gc={gc_band} band_id={band_id}: {len(seq)}!={tlen}"
                        designs.append(
                            {
                                "target_id": tid,
                                "ribo_target_index": ribo_idx,
                                "target_gc_band": float(gc_band),
                                "seed": seed,
                                "candidate_id": int(r["p_id"]),
                                "designed_sequence": seq,
                                "length": tlen,
                                "target_structure": targets[tid]["target_structure"],
                                "target_sequence": targets[tid]["target_sequence"],
                            }
                        )
    return write_folder(method, "riboswitch", targets, designs)


def seeds(pattern):
    paths = sorted(glob.glob(pattern))
    return [(seed_of(p), p) for p in paths]


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = []

    # ---- RNAinformer (plain) ------------------------------------------------
    summary.append(
        export_syn(
            "RNAinformer",
            "syn_ns",
            seeds("runs/syn_ns/version_*/predictions/syn_ns_test_preds.plk.gz"),
            "data/syn_ns/syn_ns_test.pt",
            "seq",
            gc_controlled=False,
        )
    )
    summary.append(
        export_syn(
            "RNAinformer",
            "syn_hk",
            seeds("runs/syn_hk/version_*/predictions/syn_hk_test_preds_structs.plk.gz"),
            "data/syn_hk/syn_hk_test.pt",
            "seq_db",
            gc_controlled=False,
        )
    )
    for ts in ["pdb_ts1", "pdb_ts2", "pdb_ts3", "pdb_ts_hard"]:
        summary.append(
            export_syn(
                "RNAinformer",
                ts,
                seeds(
                    f"runs/syn_pdb/version_*/predictions/{ts}_test_preds_structs.plk.gz"
                ),
                f"data/syn_pdb/{ts}_test.pt",
                "seq_bp",
                gc_controlled=False,
                filter_len=False,
            )
        )
    summary.append(
        export_ribo(
            "RNAinformer",
            seeds("runs/ribo/version_*/predictions_20/ribo_outputs.plk.gz"),
            gc=False,
        )
    )

    # ---- RNAinformer (GC-controlled) ---------------------------------------
    summary.append(
        export_syn(
            "RNAinformer_GC",
            "syn_ns",
            seeds("runs/syn_ns_gc/version_*/predictions*/gc/syn_ns_test_preds.plk.gz"),
            "data/syn_ns/syn_ns_test.pt",
            "seq",
            gc_controlled=True,
        )
    )
    summary.append(
        export_syn(
            "RNAinformer_GC",
            "syn_hk",
            seeds(
                "runs/syn_hk_gc/version_*/predictions/gc/syn_hk_test_preds_structs.plk.gz"
            ),
            "data/syn_hk/syn_hk_test.pt",
            "seq_db",
            gc_controlled=True,
        )
    )
    for ts in ["pdb_ts1", "pdb_ts2", "pdb_ts3", "pdb_ts_hard"]:
        summary.append(
            export_syn(
                "RNAinformer_GC",
                ts,
                seeds(
                    f"runs/syn_pdb_gc/version_*/predictions/gc/{ts}_test_preds_structs.plk.gz"
                ),
                f"data/syn_pdb/{ts}_test.pt",
                "seq_bp",
                gc_controlled=True,
                filter_len=False,
            )
        )
    summary.append(
        export_ribo(
            "RNAinformer_GC",
            seeds("runs/ribo_gc/version_*/predictions_20/gc/ribo_outputs_gc.plk.gz"),
            gc=True,
        )
    )

    # ---- Baselines on syn_ns (LEARNA suite + SAMFEO) -----------------------
    learna = [
        ("LEARNA", "runs/learna_suite/learna/preds.plk.gz", False),
        ("libLEARNA", "runs/learna_suite/liblearna/preds.plk.gz", False),
        ("libLEARNA_GC", "runs/learna_suite/liblearna_gc/preds.plk.gz", True),
        ("Meta-LEARNA", "runs/learna_suite/meta_learna/preds.plk.gz", False),
        (
            "Meta-LEARNA-Adapt",
            "runs/learna_suite/meta_learna_adapt/preds.plk.gz",
            False,
        ),
        ("SAMFEO", "runs/samfeo/preds.plk.gz", False),
    ]
    for name, path, gcc in learna:
        if not os.path.exists(path):
            print(f"  (skip {name}: {path} missing)")
            continue
        summary.append(
            export_syn(
                name,
                "syn_ns",
                [(0, path)],
                "data/syn_ns/syn_ns_test_learna.pt",
                "seq",
                gc_controlled=gcc,
                filter_len=False,
            )
        )

    # ---- antaRNA on syn_hk -------------------------------------------------
    summary.append(export_antarna("antaRNA", "runs/antarna/preds.plk.gz", False))
    summary.append(export_antarna("antaRNA_GC", "runs/antarna/gc/preds.plk.gz", True))

    man = pd.DataFrame(summary)
    man.to_csv(os.path.join(OUT_ROOT, "MANIFEST.csv"), index=False)
    print("\n=== MANIFEST ===")
    print(man.to_string(index=False))
    print(
        f"\nTotal designs: {man['n_designs'].sum():,} across "
        f"{len(man)} (method, dataset) folders."
    )


if __name__ == "__main__":
    main()
