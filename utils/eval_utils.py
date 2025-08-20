import numpy as np
import collections
import argparse
from collections import defaultdict
from scipy import signal
import torch
import matplotlib.pyplot as plt


def db2pairs(structure, start_index=0):
    """
    Converts dot-bracket string into a list of pairs.

    Input:
      structure <string>: A sequence in dot-bracket format.
      start_index <int>: Starting index of first nucleotide (default zero-indexing).

    Returns:
      pairs <list>: A list of tuples of (index1, index2, pk_level).

    """
    level_stacks = collections.defaultdict(list)
    closing_partners = {')': '(', ']': '[', '}': '{', '>': '<'}
    levels = {')': 0, ']': 1, '}': 2, '>': 3}

    pairs = []

    for i, sym in enumerate(structure, start_index):
        if sym == '.':
            continue
        # high order pks are alphabetical characters
        if sym.isalpha():
            if sym.isupper():
                level_stacks[sym].append(i)
            else:
                try:  # in case we have invalid preditions, we continue with next bracket
                    op = level_stacks[sym.upper()].pop()
                    pairs.append((op, i,
                                  ord(sym.upper()) - 61))  # use asci code if letter is used to asign PKs, start with level 4 (A has asci code 65)
                except:
                    continue
        else:
            if sym in closing_partners.values():
                level_stacks[sym].append(i)
            else:
                try:  # in case we have invalid preditions, we continue with next bracket
                    op = level_stacks[closing_partners[sym]].pop()
                    pairs.append([op, i, levels[sym]])
                except:
                    continue
    return sorted(pairs, key=lambda x: x[0])


def pairs2mat(pairs, length, symmetric=True, no_pk=False):
    """
    Convert list of pairs to matrix representation of structure.
    """
    # print(pairs)
    mat = np.zeros((length, length))
    if no_pk:
        for p1, p2 in pairs:
            mat[p1, p2] = 1
            if symmetric:
                mat[p2, p1] = 1
        return mat
    for p1, p2, _ in pairs:
        mat[p1, p2] = 1
        if symmetric:
            mat[p2, p1] = 1
    return mat


def db2mat(db):
    """
    Convert dot-bracket string to matrix representation of structure.
    """
    length = len(db)
    pairs = db2pairs(db)
    return pairs2mat(pairs, length)

def analyse_pairs(mat, sequence):
    pairs = mat2pairs(mat)
    wc = ['AU', 'UA', 'GC', 'CG']
    wobble = ['GU', 'UG']
    canonical = wc + wobble
    pair_types = collections.defaultdict(list)
    per_position_pairs = collections.defaultdict(list)
    closers = []
    multiplet_pairs = []
    for i, (p1, p2) in enumerate(pairs):
        pair_types['all'].append((p1, p2))
        closers.append(p2)
        per_position_pairs[p1].append((p1, p2))
        per_position_pairs[p2].append((p1, p2))
        p_type = sequence[p1] + sequence[p2]
        pair_types[''.join(sorted(p_type))].append((p1, p2))
        if p_type in wc:
            pair_types['wc'].append((p1, p2))
            pair_types['canonical'].append((p1, p2))
        elif p_type in wobble:
            pair_types['wobble'].append((p1, p2))
            pair_types['canonical'].append((p1, p2))
        else:
            pair_types['nc'].append((p1, p2))
        if i > 0 and closers[i - 1] < p2:
            pair_types['pk'].append((p1, p2))
        if len(per_position_pairs[p1]) > 1:
            multiplet_pairs += per_position_pairs[p1]
        if len(per_position_pairs[p2]) > 1:
            multiplet_pairs += per_position_pairs[p2]
    pair_types['multiplets'] = list(set(multiplet_pairs))
    pair_ratios = {k: (len(v) / len(sequence)) if not k == 'pk' else len(v) > 0 for k, v in pair_types.items()}
    return pair_ratios


def mat2pairs(matrix, symmetric=True):
    """
    Convert matrix representation of structure to list of pairs.
    """
    if symmetric:
        return list(set(tuple(sorted(pair)) for pair in np.argwhere(matrix == 1)))
    else:
        return list(tuple(pair) for pair in np.argwhere(matrix == 1))


################################################################################
# Metrics
################################################################################
def f1(tp, fp, tn, fn):
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1_score


def recall(tp, fp, tn, fn):
    recall = tp / (tp + fn + 1e-8)
    return recall


def specificity(tp, fp, tn, fn):
    specificity = tn / (tn + fp + 1e-8)
    return specificity


def precision(tp, fp, tn, fn):
    precision = tp / (tp + fp + 1e-8)
    return precision


def mcc(tp, fp, tn, fn):
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    return mcc


def non_correct(tp, fp, tn, fn):
    non_correct = (tp == 0).astype(int)
    return non_correct


def tp_from_matrices(pred, true):
    tp = np.logical_and(pred, true).sum()
    return tp


def tn_from_matrices(pred, true):
    tn = np.logical_and(np.logical_not(pred), np.logical_not(true)).sum()
    return tn


def get_fp(pred, tp):
    fp = pred.sum() - tp
    return fp


def get_fn(true, tp):
    fn = true.sum() - tp
    return fn


def solved_from_mat(pred, true):
    true = np.triu(true, k=1)
    pred = np.triu(pred, k=1)
    hits = np.logical_and(pred, true).sum()
    gt = true.sum()
    eq = np.equal(true, pred)
    acc = np.sum(np.triu(eq,k=1))/(true.shape[0]*(true.shape[0]-1)/2)
    solved = np.all(eq).astype(int)
    return solved, hits, gt, acc


def mat_from_pos(pos1, pos2, length):
    mat = np.zeros((length, length))
    for i, j in zip(pos1, pos2):
        mat[i, j] = 1
        mat[j, i] = 1
    return mat

def eval_prediction(sequence, pred_mat, true_mat, pk_info, dataset, plot_matrix=False):

    # Watson-Crick pairs: AU, UA, CG, GC
    wc_pairs = ['AU', 'UA', 'CG', 'GC']
    # Wobble pairs: GU, UG
    wobble_pairs = ['GU', 'UG']
    
    # Initialize masks
    wc = torch.zeros_like(true_mat)
    wobble = torch.zeros_like(true_mat)
    nc = torch.zeros_like(true_mat)
    canonicals = torch.zeros_like(true_mat)
    pk = torch.zeros_like(true_mat)
    multi = torch.zeros_like(true_mat)
    
    # Get positions of true base pairs
    triu_true_mat = torch.triu(torch.ones_like(true_mat), diagonal=1).bool()
    filtered_true_mat = true_mat * triu_true_mat
    true_pair_positions = torch.nonzero(filtered_true_mat, as_tuple=True)

    # print(true_pair_positions)
    multi_pred = []
    for i in range(len(sequence)):
        multi_pred.append(pred_mat[i, :].sum() > 1)
    # get specific masks for each base pair type
    for i, (p1,p2) in enumerate(zip(true_pair_positions[0], true_pair_positions[1])):

        if ''.join(sequence[p1]+sequence[p2]) in wc_pairs:
            wc[p1, p2] = 1
            wc[p2, p1] = 1
            canonicals[p1, p2] = 1
            canonicals[p2, p1] = 1
        elif ''.join(sequence[p1]+sequence[p2]) in wobble_pairs:
            wobble[p1, p2] = 1
            wobble[p2, p1] = 1
            canonicals[p1, p2] = 1
            canonicals[p2, p1] = 1
        else:
            nc[p1, p2] = 1
            nc[p2, p1] = 1

        if pk_info[i] > 0:
            pk[p1, p2] = 1
            pk[p2, p1] = 1
        
        if true_mat[p1, :].sum() > 1:
            multi[p1, p2] = 1
            multi[p2, p1] = 1
    
    if plot_matrix:
        fig, axs = plt.subplots(1, 7, figsize=(12, 4))
        axs[0].imshow(true_mat.cpu().numpy())
        axs[0].set_title("GT")
        axs[1].imshow(wc.cpu().numpy())
        axs[1].set_title("WC")
        axs[2].imshow(wobble.cpu().numpy())
        axs[2].set_title("Wobble")
        axs[3].imshow(nc.cpu().numpy())
        axs[3].set_title("NC")
        axs[4].imshow(canonicals.cpu().numpy())
        axs[4].set_title("Canonicals")
        axs[5].imshow(pk.cpu().numpy())
        axs[5].set_title("PK")
        axs[6].imshow(multi.cpu().numpy())
        axs[6].set_title("Multi")
        plt.show()
    assert (wc+wobble+nc).sum() == true_mat.sum()
    assert (canonicals+nc).sum() == true_mat.sum()

    # compute metrics
    solved = torch.equal(true_mat, pred_mat).__int__()

    num_pred_pairs = torch.sum(torch.triu(pred_mat, diagonal=1))
    num_gt_pairs = torch.sum(torch.triu(true_mat, diagonal=1))

    metrics = {'solved': int(solved),
               'num_pred_pairs': int(num_pred_pairs.cpu().item()),
               'num_gt_pairs': int(num_gt_pairs.cpu().item()),
              }

    tp = torch.logical_and(pred_mat, true_mat).sum()
    tn = torch.logical_and(torch.logical_not(pred_mat), torch.logical_not(true_mat)).sum()
    fp = torch.logical_and(pred_mat, torch.logical_not(true_mat)).sum()
    fn = torch.logical_and(torch.logical_not(pred_mat), true_mat).sum()
    metrics['num_pred_hits'] = tp.cpu().item()
    metrics['num_gt'] = true_mat.sum().cpu().item()
    assert pred_mat.size().numel() == tp + tn + fp + fn
    
   
    precision = tp / (1e-4 + tp + fp)
    recall = tp / (1e-4 + tp + fn)
    f1_score = 2 * tp / (1e-4 + (2 * tp + fp + fn))
    mcc = (tp * tn - fp * fn) / (1e-4 + torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    
    # print(sequence)
    true_labels = pred_labels = {i: sequence[i] for i in range(len(sequence))}

    wl = graph_distance_score_from_matrices(pred_mat.cpu().numpy(), true_mat.cpu().numpy(), 'WeisfeilerLehman')
    #wl_with_seq = graph_distance_score_from_matrices(pred_mat.cpu().numpy(), true_mat.cpu().numpy(), 'WeisfeilerLehman', true_labels=true_labels, pred_labels=pred_labels)

    # prec_shift, rec_shift, f1_shift = evaluate_shifted(pred_mat.cpu(), true_mat.cpu())
    
    metrics['precision'] = precision.cpu().item()
    metrics['recall'] = recall.cpu().item()
    metrics['f1_score'] = f1_score.cpu().item()
    metrics['mcc'] = mcc.cpu().item()
    metrics['wl'] = wl
    # metrics['wl_with_seq'] = wl_with_seq
    # metrics['prec_shift'] = prec_shift
    # metrics['rec_shift'] = rec_shift
    # metrics['f1_shift'] = f1_shift

    #for (name, mask) in [('wc', wc), ('wobble', wobble), ('nc', nc), ('canonicals', canonicals), ('pk', pk), ('multi', multi)]:
    for (name, mask) in [('pk', pk), ('multi', multi)]:
        if mask.sum().cpu().item() == 0:
            continue
        
        tp = torch.logical_and(pred_mat, mask).sum()
        tn = torch.logical_and(torch.logical_not(pred_mat), torch.logical_not(mask)).sum()
        fp = torch.logical_and(pred_mat, torch.logical_not(mask)).sum()
        fn = torch.logical_and(torch.logical_not(pred_mat), mask).sum()
        assert pred_mat.size().numel() == tp + tn + fp + fn

        precision = tp / (1e-4 + tp + fp)
        recall = tp / (1e-4 + tp + fn)
        f1_score = 2 * tp / (1e-4 + (2 * tp + fp + fn))
        mcc = (tp * tn - fp * fn) / (1e-4 + torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        
        metrics[f'{name}_precision'] = precision.cpu().item()
        metrics[f'{name}_recall'] = recall.cpu().item()
        metrics[f'{name}_f1_score'] = f1_score.cpu().item()
        metrics[f'{name}_mcc'] = mcc.cpu().item()

        if name == 'pk':
            metrics['pk_hit'] = tp > 0
            metrics['num_pk_hits']  = tp.cpu().item()
            metrics['num_pk_gt'] = mask.sum().cpu().item()
        if name == 'multi':
            metrics['multi_hit'] = tp > 0
            metrics['num_multi_hits']  = tp.cpu().item()
            metrics['num_multi_gt'] = mask.sum().cpu().item()
    
    return metrics

def eval_hits(pred_mat, true_mat, pk_info=None):
    pk = torch.zeros_like(true_mat)
    multi = torch.zeros_like(true_mat)
    # Get positions of true base pairs
    triu_true_mat = torch.triu(torch.ones_like(true_mat), diagonal=1).bool()
    filtered_true_mat = true_mat * triu_true_mat
    true_pair_positions = torch.nonzero(filtered_true_mat, as_tuple=True)
    
    # get specific masks for each base pair type
    for i, (p1,p2) in enumerate(zip(true_pair_positions[0], true_pair_positions[1])):
        if pk_info is not None and pk_info[i] > 0:
            pk[p1, p2] = 1
            pk[p2, p1] = 1
        if true_mat[p1, :].sum() > 1 or true_mat[p2, :].sum() > 1:
            multi[p1, p2] = 1
            multi[p2, p1] = 1
    
    metrics = {}
    for (name, mask) in [('pk', pk), ('multi', multi)]:
        if mask.sum().cpu().item() == 0:
            continue
        pred_mat = torch.triu(pred_mat, diagonal=1)
        mask = torch.triu(mask, diagonal=1)
        tp = torch.logical_and(pred_mat, mask).sum()
        tn = torch.logical_and(torch.logical_not(pred_mat), torch.logical_not(mask)).sum()
        fp = torch.logical_and(pred_mat, torch.logical_not(mask)).sum()
        fn = torch.logical_and(torch.logical_not(pred_mat), mask).sum()
        assert pred_mat.size().numel() == tp + tn + fp + fn

        precision = tp / (1e-4 + tp + fp)
        recall = tp / (1e-4 + tp + fn)
        f1_score = 2 * tp / (1e-4 + (2 * tp + fp + fn))
        mcc = (tp * tn - fp * fn) / (1e-4 + torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        
        metrics[f'{name}_precision'] = precision.cpu().item()
        metrics[f'{name}_recall'] = recall.cpu().item()
        metrics[f'{name}_f1_score'] = f1_score.cpu().item()
        metrics[f'{name}_mcc'] = mcc.cpu().item()
        if name == 'pk':
            metrics['pk_hit'] = tp > 0
            metrics['num_pk_hits']  = tp.cpu().item()
            metrics['num_pk_gt'] = mask.sum().cpu().item()
        if name == 'multi':
            metrics['multi_hit'] = tp > 0
            metrics['num_multi_hits']  = tp.cpu().item()
            metrics['num_multi_gt'] = mask.sum().cpu().item()
    return metrics

################################################################################
# from e2efold
################################################################################
# we first apply a kernel to the ground truth a
# then we multiple the kernel with the prediction, to get the TP allows shift
# then we compute f1
# we unify the input all as the symmetric matrix with 0 and 1, 1 represents pair
def shifted_f1(pred_a, true_a):
    pred_a = torch.tensor(pred_a)
    true_a = torch.tensor(true_a)

    kernel = np.array([[0.0, 1.0, 0.0],
                       [1.0, 1.0, 1.0],
                       [0.0, 1.0, 0.0]])
    pred_a_filtered = signal.convolve2d(pred_a, kernel, 'same')
    fn = len(torch.where((true_a - torch.Tensor(pred_a_filtered)) == 1)[0])
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    tp = true_p - fn
    fp = pred_p - tp
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * tp / (2 * tp + fp + fn)
    return f1_score.item()