import RNA
import numpy as np
from scipy.spatial.distance import cdist
import copy

#Code from RnaBench
################################################################################
# Riboswitch Metrics
################################################################################
hamming = lambda s1, s2: hamming_distance(s1, s2, None)

def hamming_distance(s1, s2,
                     len_mismatch_nan=True):  # TODO: include hamming distance when using 'don't care' symbol

    if len(s1) != len(s2):
        return len(s2) if not len_mismatch_nan else np.nan
    l1 = np.asarray(list(s1))
    l2 = np.asarray(list(s2))

    distance = np.sum((l1 != l2).astype(np.int8))
    return distance / len(s1)

def self_diff(df, distance, key="sequence"):
    if key is not None:
        df = df.loc[:, key]
        if key == "sequence" or key == "structure":
            ary = cdist(np.asarray(df)[..., None], np.asarray(df)[..., None],
                        metric=lambda x, y: distance(str(x[0]), str(y[0])))
        else:
            ary = cdist(np.asarray(df)[..., None], np.asarray(df)[..., None],
                        metric=distance)
        return ary
        df = df.to_numpy()
    ary = cdist(df, df,
                metric="euclidean")
    return ary

def diameter(df, distance, key="sequence", self_diff_mat=None):  # (df, distance, key="sequence"):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    return np.nanmax(self_diff_mat)


def diversity(df, distance, key="sequence", self_diff_mat=None):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    np.fill_diagonal(self_diff_mat, np.nan)
    return np.nanmean(self_diff_mat)


def sum_bottleneck(df, distance, key="sequence", self_diff_mat=None):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    np.fill_diagonal(self_diff_mat, np.nan)
    return np.sum(np.nanmin(self_diff_mat, -1))


def DPP(df, distance, key="sequence", self_diff_mat=None):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    return np.linalg.det(1 - self_diff_mat)

# is tested
def eval_shape(dot_bracket):
    return RNA.abstract_shapes(dot_bracket) == '[][]'

# is tested
def check_aptamer_structure(structure, len_aptamer=42):
    return '(((((.....)))))' in structure[:len_aptamer]

# is tested
def check_8_u(structure, length=7):
    return structure[-length:] == '.' * length

# is tested
def evaluate_co_transcriptional_folding_simulation(sequence,
                                                   aptamer,
                                                   spacer,
                                                   structure,
                                                   elongation=10):
    """
    Form the paper:
    In the current implementation, folding paths are represented as a sequence
    of secondary structures computed for sub-sequences starting at the 5′-end.
    We used individual transcription steps of 5–10 nt to simulate
    co-transcriptional folding with varying elongation speed. Secondary
    structures are computed by RNAfold, a component of the Vienna RNA Package
    (22), with parameter settings -d2 -noPS -noLP. If one of the transcription
    intermediates forms base pairs between aptamer and spacer, it is likely
    that this will interfere with the ligand-binding properties;
    hence, such a candidate is rejected.

    Changes:
    - just calling RNA.fold instead of RNAFold -d1 -noPS -noLP
    - fixed elongation speed of 10 to save compute
    """

    aptamer_len = len(aptamer)
    spacer_len = len(spacer)

    valid_intermediates = []

    for i in range(1, len(sequence), elongation):
        struc, energy = RNA.fold(sequence[:i])
        seq = sequence[:i]

        pairs_dict = db2pairs_dict_closers_keys(struc)

        if not spacer in seq:
            continue
        else:
            spacer_idx = seq.index(spacer)
            assert seq[spacer_idx:spacer_idx + spacer_len] == spacer
            if not ')' in struc[spacer_idx:spacer_idx + spacer_len]:
                valid_intermediates.append(True)
                continue
            else:
                spacer_pair_ids = []
                for i, s in enumerate(struc[spacer_idx:spacer_idx + spacer_len], spacer_idx):
                    if s == ')':
                        if pairs_dict[i] < aptamer_len - 1:
                            return False
                    valid_intermediates.append(True)
    return all(valid_intermediates)


def db2pairs_dict_closers_keys(structure):
    stack = []
    pairs = []
    for i, s in enumerate(structure):
        if s == '.':
            continue
        elif s == '(':
            stack.append(i)
        else:
            pairs.append([stack.pop(), i])
    return {p2: p1 for p1, p2 in pairs}

# is tested
def has_aptamer(seq):
    return seq[:42] == 'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA'

# is tested
def has_8_U(seq):
    return seq[-8:] == 'UUUUUUUU'

# is tested
def get_spacer_and_complement(seq, struc):
    """
    get the spacer and complement

    the spacer
    """
    stack = []
    for i, sym in enumerate(struc):
        if sym == '.':
            continue
        elif sym == '(':
            stack.append(i)
        else:
            if not stack:
                return seq[:i], seq[i:]
            stack.pop()
    return seq[:i], seq[i:]

def eval_structure(sequence, structure):
    valid = has_aptamer(sequence)
    valid_spacer = structure[42:-8].find('....') != -1 and structure[42:-8][structure[42:-8].find('....'):].find(')') != -1
    valid_8_U = has_8_U(sequence)
    valid_shape = eval_shape(structure)
    eight_u_unpaired = check_8_u(structure)
    valid_aptamer_hairpin = check_aptamer_structure(structure)

    if valid and valid_spacer and valid_8_U and valid_shape and eight_u_unpaired and valid_aptamer_hairpin:
        aptamer = sequence[:42]
        eight_u = sequence[-8:]
        spacer, complement = get_spacer_and_complement(sequence[42:-8], structure[42:-8])
        valid_co_fold = evaluate_co_transcriptional_folding_simulation(sequence,
                                                  aptamer,
                                                  spacer,
                                                  structure,
                                                  ) 
    else:
        valid_co_fold = 0
    
    results = {"valid": valid,
                "valid_spacer": valid_spacer,
                "valid_8_U": valid_8_U,
                "valid_shape": valid_shape,
                "eight_u_unpaired": eight_u_unpaired,
                "valid_aptamer_hairpin": valid_aptamer_hairpin,
                "valid_co_fold": valid_co_fold
                }
    
    results['valid_sequence_and_structure'] = all([
            valid,
            valid_spacer,
            valid_8_U,
            valid_shape,
            valid_co_fold,
            eight_u_unpaired,
            valid_aptamer_hairpin,
        ])
    
    return results

def eval_metrics(df):
    seq_dif_mat = self_diff(df, hamming, key="sequence")
    struct_dif_mat = self_diff(df, hamming, key="structure")
    results = {"seq_diameter": diameter(df, hamming, key="sequence", self_diff_mat=copy.deepcopy(seq_dif_mat)),
               "seq_diversity": diversity(df, hamming, key="sequence", self_diff_mat=copy.deepcopy(seq_dif_mat)),
                "seq_sum_bottleneck": sum_bottleneck(df, hamming, key="sequence", self_diff_mat=copy.deepcopy(seq_dif_mat)),
                "seq_DPP": DPP(df, hamming, key="sequence", self_diff_mat=copy.deepcopy(seq_dif_mat)),
                "struct_diameter": diameter(df, hamming, key="structure", self_diff_mat=copy.deepcopy(struct_dif_mat)),
                "struct_diversity": diversity(df, hamming, key="structure", self_diff_mat=copy.deepcopy(struct_dif_mat)),
                "struct_sum_bottleneck": sum_bottleneck(df, hamming, key="structure", self_diff_mat=copy.deepcopy(struct_dif_mat)),
                "struct_DPP": DPP(df, hamming, key="structure", self_diff_mat=copy.deepcopy(struct_dif_mat)),
    }
    return results