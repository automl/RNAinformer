from .ribo_metrics import eval_structure, eval_metrics
from sklearn.metrics import pairwise_distances
from RNA import fold
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

seq_vocab = ['A', 'C', 'G', 'U', 'N']
struct_vocab = ['.','(', ')']
seq_itos = dict(zip(range(1,len(seq_vocab)+1), seq_vocab))
struct_itos = dict(zip(range(len(struct_vocab)), struct_vocab))

def eval_ribo(preds, path, gc_target=None, e_target=None):
    gc_tolerance = 0.01
    energy_tolerance = 0.1 
    task_results = []
    gcs=[]
    solved_dist = []
    distances = []
    solved_unique = []
    unique = []
    gc_score = []
    for i in tqdm(range(len(preds))):
        results = []
        solved_seqs = []
        seqs = []
        for pred in preds[i]:
            seqs.append(pred)
            seq = "".join(list(map(seq_itos.get, pred)))
            struct, energy = fold(seq)
            result = eval_structure(seq, struct)
            result['energy'] = energy
            gc = (seq.count('G') + seq.count('C')) / len(seq)
            result['gc_content'] = gc
            result['structure'] = struct
            result['sequence'] = seq
            result['length'] = len(seq)
            if gc_target is not None:
                if abs(gc - gc_target) <= gc_tolerance:
                    result['valid_gc'] = True
                    if result['valid_sequence_and_structure']:
                        solved_seqs.append(pred)
                else:
                    result['valid_gc'] = False
                result['gc_score'] = 1 - abs(gc - gc_target)
                gc_score.append(result['gc_score'])
            else:
                if result['valid_sequence_and_structure']:
                    solved_seqs.append(pred)
            if e_target is not None:
                if abs(energy - e_target) <= energy_tolerance:
                    result['valid_energy'] = True
                else:
                    result['valid_energy'] = False
                result['energy_score'] = 1 - abs(energy - e_target)
            results.append(result)
        results = pd.DataFrame(results)
        task_result = eval_metrics(results.drop_duplicates(subset=['sequence']))
        metric_keys = list(task_result.keys())
        task_result['energy'] = results['energy'].mean()
        task_result['gc_content'] = results['gc_content'].mean()
        for k in metric_keys:
            task_result[f"valid_{k}"] = 0
        task_result['unique_candidates'] = len(results['sequence'].unique())
        task_result['unique_candidates_score'] = len(results['sequence'].unique())/len(results)
        if gc_target is not None:
            task_result['gc_score'] = results['gc_score'].mean()
        if e_target is not None:
            task_result['energy_score'] = results['energy_score'].mean()
        solved = results[results['valid_sequence_and_structure'] == True]
        if len(solved) > 0:
            solved_metrics = eval_metrics(solved.drop_duplicates(subset=['sequence']))
            for k, v in solved_metrics.items():
                task_result[f"valid_{k}"] = v
            task_result["unique_structures"] = len(solved['structure'].unique())
            task_result["unique_structures_score"] = len(solved['structure'].unique())/len(solved)
            task_result["unique_valid_candidates"] = len(solved['sequence'].unique())
            task_result["unique_valid_candidates_score"] = len(solved['sequence'].unique())/len(solved)
            task_result["valid_candidates_score"] = len(solved)/len(results['sequence'].unique())
            task_result["valid_candidates"] = len(solved)
            task_result['task_solved_score'] = 1
            task_result['valid_mean_energy'] = solved['energy'].mean()
            task_result['valid_mean_gc'] = solved['gc_content'].mean()
            if gc_target is not None:
                task_result['valid_gc'] = len(solved[solved['valid_gc'] == True])
                task_result['solved_gc_score'] = solved['gc_score'].mean()
            if e_target is not None:
                task_result['valid_energy'] = len(solved[solved['valid_energy'] == True])
                task_result['solved_energy_score'] = solved['energy_score'].mean()
        else:
            task_result["unique_structures"] = 0
            task_result["unique_structures_score"] = 0
            task_result["unique_valid_candidates"] = 0
            task_result["valid_candidates"] = 0
            task_result["unique_valid_candidates_score"] = 0
            task_result["valid_candidates_score"] = 0
            task_result['task_solved_score'] = 0
            if gc_target is not None:
                task_result['valid_gc'] = 0
                task_result['solved_gc_score'] = 0
            if e_target is not None:
                task_result['valid_energy'] = 0
                task_result['solved_energy_score'] = 0
        seqs = np.array(seqs)
        seqs = np.unique(seqs,axis=0)
        unique.append(len(seqs))
        dist = pairwise_distances(seqs,metric='hamming')
        masked_dist = np.ma.masked_equal(np.triu(dist),0)
        distances.append(np.mean(masked_dist))
        if len(solved_seqs)>2:
            solved_seqs = np.array(solved_seqs)
            solved_seqs = np.unique(solved_seqs,axis=0)
            solved_unique.append(len(solved_seqs))
            sdist = pairwise_distances(solved_seqs,metric='hamming')
            masked_sdist = np.ma.masked_equal(np.triu(sdist),0)
            solved_dist.append(np.mean(masked_sdist))    
        task_result['length'] = results['length'].mean()
        task_result['candidates'] = len(results)
        task_results.append(task_result)
        gcs.extend(results['gc_content'].values)
    distances = np.nan_to_num(np.array(distances, dtype=np.float32))
    solved_dist = np.nan_to_num(np.array(solved_dist, dtype=np.float32))
    task_results = pd.DataFrame(task_results)
    tasks_solved = task_results[task_results['task_solved_score'] == 1]
    if gc_target is not None:
        tasks_solved = tasks_solved[tasks_solved["valid_gc"] > 0]
    print("*****************************")
    print(f'Number of tasks solved: {len(tasks_solved)}')
    print(f'Total number of tasks: {len(task_results)}')
    print(f'Solved score: {len(tasks_solved)/len(task_results)}')
    print(f'Valid sequences: {np.mean(solved_unique)/len(preds[0])}')
    print(f'Candidate sequences Diversity: {np.mean(distances)}')
    print(f'Valid sequences Diversity: {np.mean(solved_dist)}')
    if gc_target is not None:
        print(f'GC-content error: {1-tasks_solved["gc_score"].mean()}')
    
    metrics = defaultdict(list)
    metrics['solved'] = len(tasks_solved)
    metrics['solved_score'] = len(tasks_solved)/len(task_results)
    metrics['diversity'] = np.mean(distances)
    metrics['solved_diversity'] = np.mean(solved_dist)
    metrics['unique_seq'] = np.mean(unique)
    metrics['unique_seq_score'] = np.mean(unique)/len(preds[0])
    metrics['unique_solved_seq'] = np.mean(solved_unique)
    metrics['unique_solved_seq_score'] = np.mean(solved_unique)/len(preds[0])
    metrics['gc_error'] = np.mean(gc_score)
    return metrics