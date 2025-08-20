import os
import pathlib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pandas as pd

from RNAinformer.utils.data.rna import IndexDataset, TokenBasedRandomSampler, CollatorRiboDesign, CollatorRiboDesignMat

IGNORE_INDEX = -100
PAD_INDEX = 0


class DataModuleRibo(pl.LightningDataModule):

    def __init__(
            self,
            train_dataframe_path,
            valid_dataframe_path,
            num_cpu_worker,
            num_gpu_worker,
            min_len,
            max_len,
            seed,
            batch_size,
            batch_by_token_size,
            batch_token_size,
            shuffle_pool_size,
            cache_dir,
            design,
            valid_sets,
            test_sets,
            logger,
            matrix_collate=False,
        ):
        super().__init__()
        if not os.path.exists(train_dataframe_path):
            raise UserWarning(f"dataframe does not exist: {train_dataframe_path}")
        if not os.path.exists(valid_dataframe_path):
            raise UserWarning(f"dataframe does not exist: {valid_dataframe_path}")
        self.dataframe_path = train_dataframe_path
        self.valid_dataframe_path = valid_dataframe_path
        
        if isinstance(num_gpu_worker, str):
            num_gpu_worker = len(list(map(int, num_gpu_worker.split(","))))

        self.num_gpu_worker = num_gpu_worker
        if num_cpu_worker is None:
            num_cpu_worker = os.cpu_count()
        self.num_cpu_worker = num_cpu_worker

        self.resume_index = None  # TODO not implemented yet

        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.batch_token_size = batch_token_size
        self.shuffle_pool_size = shuffle_pool_size

        self.batch_size = batch_size
        self.batch_by_token_size = batch_by_token_size

        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed
        self.design = design

        self.samples_cache_file = f"{train_dataframe_path.split('/')[-1].split('.')[0]}_len{self.min_len}{self.max_len}_design{design}_seed{self.seed}_v3.pth"
        self.logger = logger

        self.ignore_index = IGNORE_INDEX
        self.pad_index = PAD_INDEX

        self.seq_vocab = ['A', 'C', 'G', 'U', 'N']

        if design:
            self.seq_vocab = ['BOS'] + self.seq_vocab

        self.canonical_pairs = ['GC', 'CG', 'AU', 'UA', 'GU', 'UG']
        self.seq_stoi = dict(zip(self.seq_vocab, range(len(self.seq_vocab))))
        self.seq_itos = dict((y, x) for x, y in self.seq_stoi.items())
        
        self.struct_vocab = ['.', '(', ')','N']

        self.struct_itos = dict(zip(range(len(self.struct_vocab)), self.struct_vocab))
        self.struct_stoi = dict((y, x) for x, y in self.struct_itos.items())

        self.seq_vocab_size = len(self.seq_vocab)
        self.struct_vocab_size = len(self.struct_vocab)
        self.seq_mask_i = self.seq_stoi.get('N')
        self.struct_mask_i = self.struct_stoi.get('N')
        self.rng = np.random.RandomState(self.seed)
        if matrix_collate:
            self.collator = CollatorRiboDesignMat(self.pad_index, self.ignore_index)
        else:
            self.collator = CollatorRiboDesign(self.pad_index, self.ignore_index)

        self.valid_sets = valid_sets
        self.test_sets = test_sets
        self.n_samples = None

    def prepare_data(self):
        self.logger.info("Preparing data")
        if not os.path.exists(self.cache_dir / self.samples_cache_file):
            self.logger.info(
                f"Checked preprocessed data: {(self.cache_dir / self.samples_cache_file).as_posix()} does not exist.")
            self.logger.info("Start prepare data")

            train_df = pd.read_pickle(self.dataframe_path,compression='tar')
            self.logger.info(f'Finished loading dataframe (shape: {train_df.shape})')

            train_df = train_df[train_df['target_sequence'].apply(lambda x: self.min_len <= len(x) <= self.max_len)]
            train_df = train_df.reset_index()
            train_samples = []
            for id, sample in train_df.iterrows():
                sample = self._prepare_RNA_sample(sample)
                train_samples.append(sample)

            if self.n_samples is not None:
                samples_dict = {"train": train_samples[:self.n_samples]}
            else:    
                samples_dict = {"train": train_samples}
            self.logger.info(f'Finished preprocessing {len(train_samples)} train samples')

            if self.valid_sets:
                valid_df = pd.read_pickle(self.valid_dataframe_path)
                self.logger.info(f'Finished loading dataframe (shape: {valid_df.shape})')
                valid_df = valid_df[valid_df['target_sequence'].apply(lambda x: self.min_len <= len(x) <= self.max_len)]
                valid_df = valid_df.reset_index()
                valid_samples = []
                for id, sample in valid_df.iterrows():
                    sample = self._prepare_RNA_sample(sample)
                    valid_samples.append(sample)
                samples_dict['valid'] = valid_samples
                self.logger.info(f'Finished preprocessing {len(valid_samples)} valid samples')
            
            torch.save(samples_dict, self.cache_dir / self.samples_cache_file)
            self.logger.info('Dumped samples.')
        else:
            self.logger.info("Loaded from Cache")
    @staticmethod
    def sequence2index_vector(sequence, mapping):
        int_sequence = list(map(mapping.get, sequence))
        return torch.LongTensor(int_sequence)

    
    def _prepare_RNA_sample(self, input_sample):

        torch_sample = {}
        keys = input_sample.keys()
        src_seq = input_sample["target_sequence"]
        src_struct = input_sample["target_structure"] 
        trg_seq = input_sample["sequence"] if "sequence" in keys else None
        trg_struct = input_sample["structure"] if "structure" in keys else None
        gc_content = input_sample["target_gc"] if "target_gc" in keys else 0.0
        energy = input_sample["target_energy"] if "target_energy" in keys else 0.0
        length = len(src_seq)
        assert len(src_seq) == len(src_struct) == length
        if trg_seq is not None:
            assert len(trg_seq) == len(trg_struct) == length
            trg_struct = self.sequence2index_vector(trg_struct, self.struct_stoi)
            if self.design:
                trg_seq = ['BOS'] + list(trg_seq)
            trg_seq = self.sequence2index_vector(trg_seq, self.seq_stoi)
            torch_sample['trg_seq'] = trg_seq.clone()
            torch_sample['trg_struct'] = trg_struct.clone()


        src_struct = self.sequence2index_vector(src_struct, self.struct_stoi)
        src_seq = self.sequence2index_vector(src_seq, self.seq_stoi)
        seq_mask = src_seq != self.seq_mask_i
        struct_mask = src_struct != self.struct_mask_i
        torch_sample['seq_mask'] = seq_mask.to(torch.long)
        torch_sample['struct_mask'] = struct_mask.to(torch.long)
        torch_sample['src_seq'] = src_seq.clone()
        torch_sample['src_struct'] = src_struct.clone()
        torch_sample['length'] = torch.LongTensor([length])[0]
        torch_sample['gc_content'] = torch.FloatTensor([gc_content])[0]
        torch_sample['energy'] = torch.FloatTensor([energy])[0]
        
        return torch_sample

    def setup(self, stage):

        #local_rank = torch.distributed.get_rank()
        local_rank = 0
        sample_dict = torch.load(self.cache_dir / self.samples_cache_file)
        self.logger.info(
            f"Load preprocessed data from {(self.cache_dir / self.samples_cache_file).as_posix()} at rank {local_rank}.")

        for set_name, set in sample_dict.items():
            self.logger.info(f'Load preprocessed {set_name} {len(set)} samples')

        self.train_samples = sample_dict['train']
        train_indexes = list(range(len(self.train_samples)))
        train_index_dataset = IndexDataset(train_indexes)
        token_key_fn = lambda s: len(self.train_samples[s]['src_struct'])
        self.minibatch_sampler = TokenBasedRandomSampler(train_index_dataset,
                                                         token_key_fn,
                                                         batch_token_size=self.batch_token_size,
                                                         batching=True,
                                                         repeat=False,
                                                         sort_samples=True,
                                                         shuffle=True,
                                                         shuffle_pool_size=self.shuffle_pool_size,
                                                         drop_last=True,
                                                         seed=self.seed
                                                         )

        self.valid_minibatch_sampler = {}
        self.valid_samples_dict = {}
        for valid_name in self.valid_sets:
            valid_samples = sample_dict[valid_name]
            self.valid_samples_dict[valid_name] = valid_samples
            valid_indexes = list(range(len(valid_samples)))
            valid_index_dataset = IndexDataset(valid_indexes)
            valid_token_key_fn = lambda s: len(valid_samples[s]['src_struct'])
            val_sampler = TokenBasedRandomSampler(valid_index_dataset,
                                                  valid_token_key_fn,
                                                  batch_token_size=self.batch_token_size,
                                                  batching=True,
                                                  repeat=False,
                                                  sort_samples=True,
                                                  shuffle=False,
                                                  shuffle_pool_size=self.shuffle_pool_size,
                                                  drop_last=False,
                                                  seed=self.seed
                                                  )
            self.valid_minibatch_sampler[valid_name] = val_sampler

    def train_dataloader(self):
        """This will be run every epoch."""

        if self.batch_by_token_size:
            minibatches = self.minibatch_sampler.precompute_minibatches()

            if self.num_gpu_worker > 1:
                local_rank = torch.distributed.get_rank()
                numb_batches = len(minibatches)
                batches_per_rank = numb_batches // self.num_gpu_worker

                minibatches = minibatches[local_rank * batches_per_rank: (local_rank + 1) * batches_per_rank]

            def train_pl_collate_fn(indices):
                indices = minibatches[indices[0]]
                if self.partial_training:
                    raw_samples = [self.partial_sample(self.train_samples[i]) for i in indices]
                else:
                    raw_samples = [self.train_samples[i] for i in indices]

                batch = self.collator(raw_samples)

                if self.random_ignore_mat:
                    batch = self.ignore_partial_mat(batch)

                return batch

            train_indexes = list(range(len(minibatches)))
            train_index_dataset = IndexDataset(train_indexes)

            loader = DataLoader(
                train_index_dataset,
                batch_size=1,
                collate_fn=train_pl_collate_fn,
                num_workers=self.num_cpu_worker,
                pin_memory=False,
                drop_last=False,
            )

        else:
            def train_pl_collate_fn(indices):

                raw_samples = [self.train_samples[i] for i in indices]

                batch = self.collator(raw_samples)

                return batch

            train_indexes = list(range(len(self.train_samples)))
            train_indexes = self.rng.permutation(train_indexes)

            if self.num_gpu_worker > 1:
                local_rank = torch.distributed.get_rank()
                numb_samples = len(train_indexes)
                batches_per_rank = numb_samples // self.num_gpu_worker

                if numb_samples != self.num_gpu_worker * batches_per_rank:
                    train_indexes = train_indexes[: numb_samples - numb_samples % self.num_gpu_worker]

                train_indexes = train_indexes[local_rank * batches_per_rank: (local_rank + 1) * batches_per_rank]

            train_index_dataset = IndexDataset(train_indexes)

            loader = DataLoader(
                train_index_dataset,
                batch_size=self.batch_size,
                collate_fn=train_pl_collate_fn,
                num_workers=self.num_cpu_worker,
                shuffle=True,
                pin_memory=False,
                drop_last=True,
            )

        self.logger.info("Finished loading training data")
        return loader

    def val_dataloader(self):

        dataloader_list = []
        for set_name in self.valid_sets:

            if set_name not in self.valid_samples_dict:
                continue

            def val_pl_collate_fn(raw_samples):
                return self.collator(raw_samples)

            val_loader = DataLoader(
                self.valid_samples_dict[set_name],
                batch_size=self.batch_size,
                collate_fn=val_pl_collate_fn,
                num_workers=self.num_cpu_worker,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
            )

            dataloader_list.append(val_loader)
        self.logger.info(f"Finished loading validation data")
        return dataloader_list
