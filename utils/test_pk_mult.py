import torch
import tqdm
import numpy as np
import sys
from sklearn.metrics import pairwise_distances
import collections
from RNAinformer.utils.eval_utils import eval_hits, db2mat, mat_from_pos, pairs2mat, solved_from_mat, f1, recall, specificity, precision, mcc, tp_from_matrices, tn_from_matrices, get_fp, get_fn, mat2pairs, shifted_f1
import copy
from glob import glob
import argparse
from collections import defaultdict
import pandas as pd

seq_vocab = ['A', 'C', 'G', 'U', 'N']
struct_vocab = ['.', '(0', ')0', '(1', ')1', '(2', ')2']
struct_vocab = ['.', '(', ')', '[', ']', '{', '}']
seq_itos = dict(zip(range(1,len(seq_vocab)+1), seq_vocab))
struct_itos = dict(zip(range(len(struct_vocab)), struct_vocab))
wc = ['AU', 'UA', 'GC', 'CG']
wobble = ['GU', 'UG']
canonical = wc + wobble

def test_fold_pk_mult(preds, seqs, test_data, ds, gc):
    try:
        assert len(preds) == len(test_data)
    except:
        print(len(preds), len(test_data))
        raise
    if type(preds[0][0]) == str:
        for i in range(len(test_data)):
            preds[i] = [db2mat(pred) for pred in preds[i]]
    elif type(preds[0][0]) == list:
        for i in range(len(test_data)):
            length = test_data[i]['length']
            preds[i] = [pairs2mat(pred, length, no_pk=True) for pred in preds[i]]
    ds_metrics = collections.defaultdict(list)
    ns_metrics = collections.defaultdict(list)
    pk_metrics = collections.defaultdict(list)
    multi_metrics = collections.defaultdict(list)
    total = len(test_data)
    total_ns = 0
    total_pk = 0
    total_multiplets = 0
    solved_seq = [[] for i in range(len(test_data))]
    pk_solved_seq = [[] for i in range(len(test_data))]
    multiplet_solved_seq = [[] for i in range(len(test_data))]
    solved = np.zeros(len(test_data))
    gc_solved = np.zeros(len(test_data))
    ns_solved = np.zeros(len(test_data))
    ns_nc_solved = np.zeros(len(test_data))
    pk_solved = np.zeros(len(test_data))
    pk_nc_solved = np.zeros(len(test_data))
    nc_solved=np.zeros(len(test_data))
    nc_gc_solved = np.zeros(len(test_data))
    multiplet_solved = np.zeros(len(test_data))
    multiplet_nc_solved = np.zeros(len(test_data))
    gc_tolerance = 0.01
    gc_scores = []
    gc_avg_scores = []
    lengths = []
    gc_targets = []
    solutions = []
    for i in tqdm.tqdm(range(len(preds))):
        has_pk = test_data[i]['has_pk']
        has_multiplet = test_data[i]['has_multiplet']
        metrics = collections.defaultdict(list)
        struct = test_data[i]['src_struct']
        struct = list(map(struct_itos.get, struct.tolist()))
        length = test_data[i]['length']
        seq = test_data[i]['trg_seq'][1:]
        seq = "".join(list(map(seq_itos.get, seq.tolist())))
        if gc:
            gc_content = test_data[i]['gc_content']
            gc_targets.append(gc_content.item())
        assert preds[i][0].shape == (length, length)
        lengths.append(length.item())
        true_mat = mat_from_pos(test_data[i]['pos1id'], test_data[i]['pos2id'], length)
        for j, pred in enumerate(preds[i]):
            pk_info = test_data[i]['pk']
            str_acc, hits, gt, acc = solved_from_mat(pred, true_mat)
            tp = tp_from_matrices(pred, true_mat)
            tn = tn_from_matrices(pred, true_mat)
            fp = get_fp(pred, tp)
            fn = get_fn(true_mat, tp)
            metrics['f1'].append(f1(tp, fp, tn, fn))
            metrics['recall'].append(recall(tp, fp, tn, fn))
            metrics['specificity'].append(specificity(tp, fp, tn, fn))
            metrics['precision'].append(precision(tp, fp, tn, fn))
            metrics['mcc'].append(mcc(tp, fp, tn, fn))
            metrics['shifted_f1'].append(shifted_f1(pred, true_mat))
            metrics['acc'].append(acc)
            pred_gc = (list(seqs[i][j]).count(2) + list(seqs[i][j]).count(3))/length
            if gc:
                gc_score = abs(pred_gc - gc_content)
                gc_avg_scores.append(1-gc_score)
                metrics['gc_score'].append(1-gc_score)
            if str_acc == 1:
                solved[i]+=1
                seq_nc=0
                pairs = mat2pairs(pred)
                pred_seq = list(map(seq_itos.get, seqs[i][j]))
                for p1, p2 in pairs:
                    if pred_seq[p1] + pred_seq[p2] not in canonical:
                        nc_solved[i] += 1
                        seq_nc = 1
                        break
                if gc:
                    if gc_score <= gc_tolerance:
                        gc_solved[i] += 1
                        solved_seq[i].append(seqs[i][j])
                        if seq_nc:
                            nc_gc_solved[i]+=1
                    gc_scores.append(1-gc_score)
                    metrics['gc_score_str'].append(1-gc_score)
                else:
                    solved_seq[i].append(seqs[i][j])
        for k in metrics.keys():
            ds_metrics[k].append(np.mean(metrics[k]))
            ds_metrics[f"{k}_max"].append(np.max(metrics[k]))

        if has_pk:
            total_pk += 1
            if solved[i] > 0:
                pk_solved_seq[i]=copy.deepcopy(solved_seq[i])
                pk_solved[i] = gc_solved[i] if gc else solved[i]
                pk_nc_solved[i] = nc_gc_solved[i] if gc else nc_solved[i]
            for k in metrics.keys():
                pk_metrics[k].append(np.mean(metrics[k]))
                pk_metrics[f"{k}_max"].append(np.max(metrics[k]))
        if has_multiplet:
            total_multiplets += 1
            if solved[i] > 0:
                multiplet_solved_seq[i]=copy.deepcopy(solved_seq[i])
                multiplet_solved[i] = gc_solved[i] if gc else solved[i]
                multiplet_nc_solved[i] = nc_gc_solved[i] if gc else nc_solved[i]
            for k in metrics.keys():
                multi_metrics[k].append(np.mean(metrics[k]))
                multi_metrics[f"{k}_max"].append(np.max(metrics[k]))
        if not has_pk and not has_multiplet:
            total_ns += 1
            if solved[i] > 0:
                ns_solved[i] = gc_solved[i] if gc else solved[i]
                ns_nc_solved[i] = nc_gc_solved[i] if gc else nc_solved[i]
            for k in metrics.keys():
                ns_metrics[k].append(np.mean(metrics[k]))
                ns_metrics[f"{k}_max"].append(np.max(metrics[k]))
        
    distances = []
    unique_seqs = []
    for seq in seqs:
        unique_seqs.append(len(np.unique(seq, axis=0)))
        seq = np.unique(seq, axis=0)
        dist = np.nan_to_num(pairwise_distances(np.array(seq),metric='hamming'))
        masked_dist = np.ma.masked_values(np.triu(dist),0.0)
        distances.append(masked_dist.mean())
    solved_distances=[]
    solved_unique_seqs = []
    for seq in solved_seq:
        if len(seq) == 0:
            continue
        solved_unique_seqs.append(len(np.unique(seq, axis=0)))
        if len(seq) < 2:
            continue
        seq = np.unique(seq, axis=0)
        dist = np.nan_to_num(pairwise_distances(np.array(seq),metric='hamming'))
        masked_dist = np.ma.masked_values(np.triu(dist),0.0)
        solved_distances.append(masked_dist.mean())

    pk_distances=[]
    pk_unique_seqs=[]
    if np.sum(pk_solved) > 0:
        for seq in pk_solved_seq:
            if len(seq) == 0:
                continue
            pk_unique_seqs.append(len(np.unique(seq, axis=0)))
            if len(seq) < 2:
                continue
            seq = np.unique(seq, axis=0)
            dist = np.nan_to_num(pairwise_distances(np.array(seq),metric='hamming'))
            masked_dist = np.ma.masked_values(np.triu(dist),0.0)
            pk_distances.append(masked_dist.mean())
    
    multiplet_distances=[]
    multiplet_unique_seqs=[]
    if np.sum(multiplet_solved) > 0:
        for seq in multiplet_solved_seq:
            if len(seq) == 0:
                continue
            multiplet_unique_seqs.append(len(np.unique(seq, axis=0)))
            if len(seq) < 2:
                continue
            seq = np.unique(seq, axis=0)
            dist = np.nan_to_num(pairwise_distances(np.array(seq),metric='hamming'))
            masked_dist = np.ma.masked_values(np.triu(dist),0.0)
            multiplet_distances.append(masked_dist.mean())

    ns_distances = []
    ns_unique_seqs = []
    if np.sum(ns_solved) > 0:
        for seq in solved_seq:
            if len(seq) == 0:
                continue
            ns_unique_seqs.append(len(np.unique(seq, axis=0)))
            if len(seq) < 2:
                continue
            seq = np.unique(seq, axis=0)
            dist = np.nan_to_num(pairwise_distances(np.array(seq),metric='hamming'))
            masked_dist = np.ma.masked_values(np.triu(dist),0.0)
            ns_distances.append(masked_dist.mean())
    
    distances = np.nan_to_num(np.array(distances, dtype=np.float32))
    solved_distances = np.nan_to_num(np.array(solved_distances, dtype=np.float32))
    pk_distances = np.nan_to_num(np.array(pk_distances, dtype=np.float32))
    multiplet_distances = np.nan_to_num(np.array(multiplet_distances, dtype=np.float32))
    ns_distances = np.nan_to_num(np.array(ns_distances, dtype=np.float32))
    metrics={}
    if gc:
        solved = gc_solved
        nc_solved = nc_gc_solved
    print("Number of tasks solved:", np.sum(solved>0))
    print("Total no of tasks:", total)
    print("Solved score:", np.sum(solved>0)/total)
    print("valid sequences:", np.mean(solved_unique_seqs)/len(preds[0]))
    print("Valid sequences diversity:", solved_distances.mean())
    print("Valid sequences with Non-Canonical Base Pairs:", np.sum(nc_solved)/np.sum(solved))
    print("Max F1:", np.mean(ds_metrics["f1_max"]))
    print("Max MCC:", np.mean(ds_metrics["mcc_max"]))
    print("Mean F1:", np.mean(ds_metrics["f1"]))
    print("Mean MCC:", np.mean(ds_metrics["mcc"]))
    if gc:
        print("GC-content error for structure solved:", 1-np.mean(gc_scores))
        print("GC-content error:", 1-np.mean(gc_avg_scores))
    if total_pk > 0:
        print("No of solved pk tasks:", np.sum(pk_solved>0))
        print("Total no of pk tasks:", total_pk)
        print("Pk Solved score:", np.sum(pk_solved>0)/total_pk)
        if np.sum(pk_solved) > 0:
            print("valid sequences:", np.mean(pk_unique_seqs)/len(preds[0]))
            print("Valid sequences diversity:", pk_distances.mean())
            print("Valid sequences with Non-Canonical Base Pairs:", np.sum(pk_nc_solved)/np.sum(pk_solved))
        print("pK Max F1:", np.mean(pk_metrics["f1_max"]))
        print("pK Max MCC:", np.mean(pk_metrics["mcc_max"]))
        print("pK Mean F1:", np.mean(pk_metrics["f1"]))
        print("pK Mean MCC:", np.mean(pk_metrics["mcc"]))
    if total_multiplets > 0:
        print("No of solved multiplet tasks:", np.sum(multiplet_solved>0))
        print("Total no of multiplet tasks:", total_multiplets)
        print("Multiplet Solved score:", np.sum(multiplet_solved>0)/total_multiplets)
        if np.sum(multiplet_solved) > 0:
            print("valid sequences:", np.mean(multiplet_unique_seqs)/len(preds[0]))
            print("Valid sequences diversity:", multiplet_distances.mean())
            print("Valid sequences with Non-Canonical Base Pairs:", np.sum(multiplet_nc_solved)/np.sum(multiplet_solved))
        print("Multiplet Max F1:", np.mean(multi_metrics["f1_max"]))
        print("Multiplet Max MCC:", np.mean(multi_metrics["mcc_max"]))
        print("Multiplet Mean F1:", np.mean(multi_metrics["f1"]))
        print("Multiplet Mean MCC:", np.mean(multi_metrics["mcc"]))
    if total_ns > 0:
        print("No of solved ns tasks:", np.sum(ns_solved>0))
        print("Total no of ns tasks:", total_ns)
        print("Solved score:", np.sum(ns_solved>0)/total_ns)
        if np.sum(ns_solved) > 0:
            print("valid sequences:", np.mean(ns_unique_seqs)/len(preds[0]))
            print("Valid sequences diversity:", ns_distances.mean())
            print("Valid sequences with Non-Canonical Base Pairs:", np.sum(ns_nc_solved)/np.sum(ns_solved))
    score_metrics = ['f1_max', 'recall_max', 'specificity_max', 'precision_max', 'mcc_max'] 
    metrics['valid_seq'] = np.mean(solved_unique_seqs)/len(preds[0])
    metrics['diversity'] = solved_distances.mean()
    metrics['NC'] = np.sum(nc_solved)/np.sum(solved)
    if gc:
        metrics['gc_error'] = 1-np.mean(gc_avg_scores)
    for k in ds_metrics:
        if k in score_metrics:
            metrics[k] = np.mean(ds_metrics[k])
    if total_pk > 0:
        metrics['solved_pk'] = np.sum(pk_solved>0)/total_pk
        metrics['pk_diversity'] = pk_distances.mean()
        metrics['pk_valid_seq'] = np.mean(pk_unique_seqs)/len(preds[0])
        metrics['pk_NC'] = np.sum(pk_nc_solved)/np.sum(pk_solved)
        for k in pk_metrics:
            if k in score_metrics:
                metrics[f"pk_{k}"] = np.mean(pk_metrics[k])
        if gc:
            metrics['pk_gc_error'] = 1-np.mean(pk_metrics['gc_scores'])
    if total_multiplets > 0:
        metrics['solved_multiplet'] = np.sum(multiplet_solved>0)/total_multiplets
        metrics['multiplet_diversity'] = multiplet_distances.mean()
        metrics['multiplet_valid_seq'] = np.mean(multiplet_unique_seqs)/len(preds[0])
        metrics['multiplet_NC'] = np.sum(multiplet_nc_solved)/np.sum(multiplet_solved)
        for k in multi_metrics:
            if k in score_metrics:
                metrics[f"multiplet_{k}"] = np.mean(multi_metrics[k])
        if gc:
            metrics['multiplet_gc_error'] = 1-np.mean(multi_metrics['gc_scores'])
    if total_ns > 0:
        metrics['solved_ns'] = np.sum(ns_solved>0)/total_ns
        metrics['ns_diversity'] = ns_distances.mean()
        metrics['ns_valid_seq'] = np.mean(ns_unique_seqs)/len(preds[0])
        metrics['ns_NC'] = np.sum(ns_nc_solved)/np.sum(ns_solved)
        for k in ns_metrics:
            if k in score_metrics:
                metrics[f"ns_{k}"] = np.mean(ns_metrics[k])
        if gc:
            metrics['ns_gc_error'] = 1-np.mean(ns_metrics['gc_scores'])

   
    return metrics