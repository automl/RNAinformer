from collections import defaultdict
from RNA import fold
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import tqdm
import argparse
import subprocess
import psutil
import os
from .eval_utils import f1, recall, specificity, precision, mcc , db2mat, tp_from_matrices, tn_from_matrices, get_fp, get_fn, solved_from_mat
seq_vocab = ['A', 'C', 'G', 'U', 'N']
struct_vocab = ['.','(', ')','[',']','{','}','<','>']
seq_itos = dict(zip(range(1,len(seq_vocab)+1), seq_vocab))
struct_itos = dict(zip(range(len(struct_vocab)), struct_vocab))
seq_stoi = dict(zip(seq_vocab,range(1,len(seq_vocab)+1)))
struct_stoi = dict(zip(struct_vocab,range(len(struct_vocab))))

def test_fold(test_preds,test_data, max_len=800, gc=False):
    gc_tolerance = 0.01
    distances= []
    zero_count=0
    solved=0
    solved_dist = []
    nsolved=[]
    gc_nsolved = []
    energy_nsolved = []
    ds_metrics = defaultdict(list)
    gc_solved = 0
    energy_solved = 0
    test_data = [test_data[i] for i in range(len(test_data)) if test_data[i]['length']<=max_len]
    test_preds = [test_preds[i] for i in range(len(test_preds)) if len(test_preds[i][0])<=max_len]
    skipped = 0
    gc_scores = []
    gc_avg_scores = []
    gc_values = []
    gc_targets = []
    gc_solved_values = []
    solved_unique = []
    unique = []
    print(f"Test data: {len(test_data)}")
    print(f"Test preds: {len(test_preds)}")
    assert len(test_data) == len(test_preds)
    for i,pred in tqdm.tqdm(enumerate(test_preds)):
        metrics=defaultdict(list)
        solved_seqs = []
        length = test_data[i]['length']
        struct = test_data[i]['src_struct']
        count=0
        gc_count=0
        struct = list(map(struct_itos.get, struct.tolist()))
        true_mat = db2mat(struct)
        gc_solved_value = []
        if length <= max_len:
            for seq_pred_i in pred:
                assert len(seq_pred_i) == length
                seq_pred = list(map(seq_itos.get, seq_pred_i))
                if None in seq_pred:
                    skipped+=1
                    continue
                fold_struct_pred, fold_energy_pred = fold("".join(seq_pred))
                fold_struct_pred = list(fold_struct_pred)
                pred_mat = db2mat(fold_struct_pred)
                tp = tp_from_matrices(pred_mat, true_mat)
                tn = tn_from_matrices(pred_mat, true_mat)
                fp = get_fp(pred_mat, tp)
                fn = get_fn(true_mat, tp)
                metrics['f1'].append(f1(tp, fp, tn, fn))
                metrics['recall'].append(recall(tp, fp, tn, fn))
                metrics['specificity'].append(specificity(tp, fp, tn, fn))
                metrics['precision'].append(precision(tp, fp, tn, fn))
                metrics['mcc'].append(mcc(tp, fp, tn, fn))
                str_acc, hits, gt, acc  = solved_from_mat(pred_mat, true_mat)
                gc_pred = ("".join(seq_pred).count('G') + "".join(seq_pred).count('C')) / len(seq_pred)
                gc_values.append(gc_pred)
                gc_target = test_data[i]['gc_content']
                gc_score = abs(gc_pred - gc_target)
                if str_acc:
                    count+=1
                    if gc:
                        gc_targets.append(gc_target)
                        if  gc_score <= gc_tolerance:
                            gc_count+=1
                            gc_solved_value.append(gc_pred)
                            solved_seqs.append(seq_pred_i)
                        gc_scores.append(1-gc_score)
                    else:
                        solved_seqs.append(seq_pred_i)
                if gc:
                    gc_avg_scores.append(1-gc_score)
            for k in metrics.keys():
                if len(metrics[k])>0:
                    ds_metrics[k].append(np.mean(metrics[k]))
                    ds_metrics[k+"_max"].append(np.max(metrics[k]))
            if count == 0:
                zero_count+=1
            else:
                solved+=1
                nsolved.append(count)
                if gc_count>0:
                    gc_solved+=1
                    gc_nsolved.append(gc_count)
            gc_solved_values.append(gc_solved_value)
            pred = np.array(pred).astype(np.int32)
            pred = np.unique(pred,axis=0)
            unique.append(len(pred))
            dist = pairwise_distances(pred,metric='hamming')
            masked_dist = np.ma.masked_equal(np.triu(dist),0)
            distances.append(np.mean(masked_dist))
            if len(solved_seqs)>2:
                solved_seqs = np.array(solved_seqs)
                solved_seqs = np.unique(solved_seqs,axis=0)
                solved_unique.append(len(solved_seqs))
                sdist = pairwise_distances(solved_seqs,metric='hamming')
                masked_sdist = np.ma.masked_equal(np.triu(sdist),0)
                solved_dist.append(np.mean(masked_sdist))

    total = solved+zero_count
    if gc:
        solved = gc_solved         
    distances = np.nan_to_num(np.array(distances, dtype=np.float32))
    solved_dist = np.nan_to_num(np.array(solved_dist, dtype=np.float32))
    print(f"No of solved: {solved}")
    print(f"solved%: {solved/(total)}")
    if gc:
        print(f"GC-content error for structure solved: {1-np.mean(gc_scores)}")
        print(f"GC-content error: {1-np.mean(gc_avg_scores)}")
    print(f"Valid Sequence Diversity: {np.mean(solved_dist)}")
    print(f"No of solved seq per str: {np.mean(nsolved)}")
    print(f"Valid Sequences: {np.mean(solved_unique)}")
    metrics = {}
    score_metrics = ['f1_max', 'recall_max', 'specificity_max', 'precision_max', 'mcc_max']
    for k in ds_metrics.keys():
        if k in score_metrics:
            metrics[k] = np.mean(ds_metrics[k])
    for k in metrics.keys():
        print(f"{k}: {metrics[k]}")
    metrics['solved'] = solved/(total)
    metrics['diversity'] = np.mean(solved_dist)
    metrics['valid_seq'] = np.mean(solved_unique)/len(test_preds[0])
    if gc:
        metrics['gc_error_str'] = 1-np.mean(gc_scores)
        metrics['gc_error'] = 1-np.mean(gc_avg_scores)

    return metrics 
