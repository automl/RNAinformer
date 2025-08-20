from typing import List
import torch
import numpy as np


class IndexDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)


class TokenBasedRandomSampler:

    def __init__(self, dataset, token_key_fn, batch_token_size, batching, repeat, sort_samples, shuffle,
                 shuffle_pool_size, drop_last, seed=1):

        super().__init__()
        self.token_length = [token_key_fn(s) for s in dataset]

        self.batch_token_size = batch_token_size

        self.repeat = repeat
        self.batching = batching
        self.sort_samples = sort_samples
        self.drop_last = drop_last

        self.shuffle = shuffle
        self.shuffle_pool_size = shuffle_pool_size

        self.rng = np.random.default_rng(seed=seed)

        self.reverse = False

        if not self.repeat:
            self.minibatches = self.precompute_minibatches()

    def __len__(self):
        if not self.batching:
            return len(self.token_length)
        else:
            return len(self.minibatches)

    def get_index_list(self):
        index_list = [i for i in range(len(self.token_length))]
        if self.shuffle:
            self.rng.shuffle(index_list)
        return index_list

    def get_index_iter(self):
        while True:
            index_list = self.get_index_list()
            for i in index_list:
                yield i

    def pool_and_sort(self, sample_iter):
        pool = []
        for sample in sample_iter:
            if not self.sort_samples:
                yield sample
            else:
                pool.append(sample)
                if len(pool) >= self.shuffle_pool_size:
                    pool.sort(key=lambda x: self.token_length[x], reverse=self.reverse)
                    self.reverse = not self.reverse
                    while len(pool) > 0:
                        yield pool.pop()
        if len(pool) > 0:
            pool.sort(key=lambda x: self.token_length[x], reverse=self.reverse)
            self.reverse = not self.reverse
            while len(pool) > 0:
                yield pool.pop()

    def get_minibatches(self, index_iter):

        minibatch, max_size_in_batch = [], 0

        if self.batching and self.shuffle and self.shuffle_pool_size and self.sort_samples:
            index_iter = self.pool_and_sort(index_iter)

        for sample in index_iter:

            if self.batching:
                minibatch.append(sample)
                max_size_in_batch = max(max_size_in_batch, self.token_length[sample])
                size_so_far = len(minibatch) * max(max_size_in_batch, self.token_length[sample])
                if size_so_far == self.batch_token_size:
                    yield minibatch
                    minibatch, max_size_in_batch = [], 0
                if size_so_far > self.batch_token_size:
                    yield minibatch[:-1]
                    minibatch = minibatch[-1:]
                    max_size_in_batch = self.token_length[minibatch[0]]
            else:
                yield [sample]

        if (not self.drop_last) and len(minibatch) > 0:
            yield minibatch

    def precompute_minibatches(self):
        index_iter = self.get_index_list()

        minibatches = [m for m in self.get_minibatches(index_iter) if len(m) > 0]
        if self.shuffle:
            self.rng.shuffle(minibatches)
        return minibatches

    def __iter__(self):
        if self.repeat:
            index_iter = self.get_index_iter()
            for batch in self.get_minibatches(index_iter):
                yield batch
        else:

            for m in self.minibatches:
                yield m


class CollatorRNADesignMat:
    def __init__(self, pad_index, ignore_index):
        self.ignore_index = ignore_index
        self.pad_index = pad_index

    def __call__(self, samples, neg_samples=False) -> List[List[int]]:
        # tokenize the input text samples

        with torch.no_grad():
            batch_dict = {k: [dic[k] for dic in samples] for k in samples[0] if k in ['length', 'pdb_sample', 'pos1id','energy','gc_content']}

            batch_dict['length'] = torch.stack(batch_dict['length'])

            if 'pdb_sample' in batch_dict:
                batch_dict['pdb_sample'] = torch.stack(batch_dict['pdb_sample'])
            
            if 'energy' in batch_dict:
                batch_dict['energy'] = torch.stack(batch_dict['energy'])

            if 'gc_content' in batch_dict:
                batch_dict['gc_content'] = torch.stack(batch_dict['gc_content'])

            max_len = batch_dict['length'].max()
            batch_size = len(samples)

            src_struct = torch.LongTensor(batch_size, max_len, max_len).fill_(self.pad_index)

            if 'pos1id' in batch_dict:
                max_pos = max(pos.shape[0] for pos in batch_dict['pos1id'])
                pos1id = torch.full((batch_size, max_pos), self.pad_index)
                pos2id = torch.full((batch_size, max_pos), self.pad_index)
                trg_seq = torch.full((batch_size, max_len + 1), self.ignore_index)
                
            for b_id, sample in enumerate(samples):
                if 'pos1id' in batch_dict:
                    pos1id[b_id, :sample['pos1id'].size(0)] = sample['pos1id']
                    pos2id[b_id, :sample['pos2id'].size(0)] = sample['pos2id']
                    trg_seq[b_id, :sample['trg_seq'].size(0)] = sample['trg_seq']
                    src_struct[b_id, sample['pos1id'], sample['pos2id']] = 1
                    src_struct[b_id, sample['pos2id'], sample['pos1id']] = 1
                    
            if 'pos1id' in batch_dict:
                batch_dict['pos1id'] = pos1id
                batch_dict['pos2id'] = pos2id
                batch_dict['trg_seq'] = trg_seq
                batch_dict['src_struct'] = src_struct
                
           
        return batch_dict
    

class CollatorRiboDesignMat:
    def __init__(self, pad_index, ignore_index):
        self.ignore_index = ignore_index
        self.pad_index = pad_index

    def __call__(self, samples, neg_samples=False) -> List[List[int]]:
        # tokenize the input text samples
        with torch.no_grad():
            batch_dict = {k: [dic[k] for dic in samples] for k in samples[0] if k in ['length', 'pos1id','energy','gc_content','trg_seq']}

            batch_dict['length'] = torch.stack(batch_dict['length'])

            if 'pdb_sample' in batch_dict:
                batch_dict['pdb_sample'] = torch.stack(batch_dict['pdb_sample'])
            
            if 'energy' in batch_dict:
                batch_dict['energy'] = torch.stack(batch_dict['energy'])

            if 'gc_content' in batch_dict:
                batch_dict['gc_content'] = torch.stack(batch_dict['gc_content'])

            max_len = batch_dict['length'].max()
            batch_size = len(samples)

            src_struct = torch.LongTensor(batch_size, max_len, max_len).fill_(self.pad_index)
            src_seq = torch.full((batch_size,max_len), self.pad_index)
            seq_mask = torch.full((batch_size,max_len), self.pad_index)
            struct_mask = torch.full((batch_size,max_len), self.pad_index)
            
            if 'pos1id' in batch_dict:
                max_pos = max(pos.shape[0] for pos in batch_dict['pos1id'])
                pos1id = torch.full((batch_size, max_pos), self.pad_index)
                pos2id = torch.full((batch_size, max_pos), self.pad_index)
            trg_seq = torch.full((batch_size, max_len + 1), self.ignore_index)
            trg_struct = torch.full((batch_size, max_len), self.ignore_index)
                
            for b_id, sample in enumerate(samples):
                src_seq[b_id, :sample['src_seq'].size(0)] = sample['src_seq']
                seq_mask[b_id, :sample['seq_mask'].size(0)] = sample['seq_mask']
                struct_mask[b_id, :sample['struct_mask'].size(0)] = sample['struct_mask']
                if 'pos1id' in batch_dict:
                    pos1id[b_id, :sample['pos1id'].size(0)] = sample['pos1id']
                    pos2id[b_id, :sample['pos2id'].size(0)] = sample['pos2id']
                    src_struct[b_id, sample['pos1id'], sample['pos2id']] = 1
                    src_struct[b_id, sample['pos2id'], sample['pos1id']] = 1
                else:
                    struct = torch.LongTensor(sample['src_struct'].size(0), sample['src_struct'].size(0)).fill_(2)
                    for i in range(2,-1,-1):
                        if i == 2:
                            id = 1
                        else:
                            id = i
                        struct[sample['src_struct']==i,:] = id
                        struct[:,sample['src_struct']==i] = id
                    struct.fill_diagonal_(0)
                    src_struct[b_id, :sample['src_struct'].size(0), :sample['src_struct'].size(0)] = struct
                if 'trg_seq' in sample:
                        trg_seq[b_id, :sample['trg_seq'].size(0)] = sample['trg_seq']
                        trg_struct[b_id, :sample['trg_struct'].size(0)] = sample['trg_struct']
            if 'trg_seq' in batch_dict:
                batch_dict['trg_seq'] = trg_seq
                batch_dict['trg_struct'] = trg_struct

            if 'pos1id' in batch_dict:
                batch_dict['pos1id'] = pos1id
                batch_dict['pos2id'] = pos2id
            batch_dict['src_struct'] = src_struct
            batch_dict['src_seq'] = src_seq
            batch_dict['seq_mask'] = seq_mask
            batch_dict['struct_mask'] = struct_mask
           
        return batch_dict



    
