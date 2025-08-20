from collections import defaultdict
import os
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import inspect

from RNAinformer.model.RNADesignFormer import RNADesignFormer
from RNAinformer.model.RiboDesignFormer import RiboDesignFormer
from RNAinformer.utils import instantiate
from RNAinformer.utils.group_parameters import group_parameters_for_optimizer
from RNAinformer.utils.optim.lr_schedule import get_learning_rate_schedule
from flash_attn.losses.cross_entropy import CrossEntropyLoss as flash_CrossEntropyLoss
import gc


class RNADesignTrainer(pl.LightningModule):

    def __init__(
            self,
            cfg_train,
            cfg_model,
            py_logger,
            val_sets_name,
            ignore_index,
            save_dir,
            constrained_design=False,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.cfg_train = cfg_train

        self.val_sets_name = val_sets_name
        self.ignore_index = ignore_index
        self.py_logger = py_logger
        self.constrained_design = constrained_design
        if self.constrained_design:
            self.model = RiboDesignFormer(cfg_model)
        else:
            self.model = RNADesignFormer(cfg_model)

        if self.cfg_train.get('flash_loss',True):
            self.loss_train = flash_CrossEntropyLoss(ignore_index=self.ignore_index,reduction='none',label_smoothing=0.0)
            self.loss_valid = flash_CrossEntropyLoss(ignore_index=self.ignore_index,reduction='none',label_smoothing=0.0)
        else:
            self.loss_train = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index,reduction='none',label_smoothing=0.0)
            self.loss_valid = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index,reduction='none',label_smoothing=0.0)
        
        self.intern_log = []

        self.train_outputs = []
        self.valid_outputs = []

        self.val_fold_test = cfg_train.val_fold_test
        self.n_samples = cfg_train.n_samples
        self.final_val = False
        self.generated_seq = []
        self.val_batches = []
        self.val_lengths = []
        self.save_dir = save_dir +"validation_samples"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    def on_train_start(self):
        
        if self.cfg_train.optimizer.scheduler_mult_factor is not None:
            self.py_logger.info(
                f"Multiplying all LR schedule lambas by {self.cfg_train.optimizer.scheduler_mult_factor}"
            )
            self.lr_schedulers().lr_lambdas = [
                lambda x: self.cfg_train.optimizer.scheduler_mult_factor * fn(x)
                for fn in self.lr_schedulers().lr_lambdas
            ]
    def training_step(self, batch, batch_idx): 
        metrics = defaultdict(list)
        if self.constrained_design:
            logits = self.model(batch['src_seq'], batch['src_struct'], batch['seq_mask'], batch['struct_mask'], batch['trg_seq'], batch['length'], batch['gc_content'])
        else:
            logits = self.model(batch['src_struct'], batch['trg_seq'], batch['length'], batch['gc_content'])
        target = batch['trg_seq'][:, 1:]
        loss = self.loss_train(logits.view(-1,logits.shape[-1]), target.contiguous().view(-1))
        loss = loss.view(target.shape)

        loss = loss.sum(1)/batch['length']
        perplexity = torch.exp(loss)
        metrics['epoch_perplexity'].extend(perplexity.detach().cpu())
        perplexity = perplexity.mean()
        loss = loss.mean()
        
        pred_seq = torch.argmax(logits, dim=-1)
        accuracy = torch.logical_and(pred_seq == target, target != self.ignore_index).sum(1) / batch['length']
        metrics['accuracy'].extend(accuracy.detach().cpu())
        accuracy = accuracy.mean()
        self.log(
            f"train/loss",
            loss.detach().cpu(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"train/perplexity",
            perplexity.detach().cpu(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"train/accuracy",
            accuracy.detach().cpu(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("global_step", torch.FloatTensor([self.global_step]))
        return_dict = {"mean_batch_length": torch.mean(batch['length'].to(torch.float).cpu()),
                "batch_size": batch['length'].shape[0],
                "count": batch['length'].shape[0]}
        metrics = {k: torch.stack(v).sum().cpu() for k, v in metrics.items()}
        return_dict.update(metrics)
        self.train_outputs.append(return_dict)
        return {"loss": loss}
    
    def on_train_batch_end(self,outputs, batch, batch_idx):
        torch.cuda.empty_cache()
    
    def on_train_epoch_end(self):
        outputs = self.train_outputs
        values = ["mean_batch_length", "batch_size", "count", "accuracy", "epoch_perplexity"]

        summed_values = {k: 0 for k in values}
        for out_dict in outputs:
            for key in values:
                summed_values[key] += out_dict[key]

        metrics = {"batch_length": summed_values['mean_batch_length'] / len(outputs),
                   "batch_size": summed_values['batch_size'] / len(outputs)}

        for name, value in metrics.items():
            self.log(f"train/{name}", value,
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, )
            if self.local_rank == 0:
                print(f"train/{name}", value, self.local_rank)
        
        self.train_outputs = []

    def validation_step(self, batch, batch_idx, dataset_index=None):

        metrics = defaultdict(list)
        if dataset_index is None:
            dataset_index = 0
        with torch.no_grad():
            if self.constrained_design:
                logits = self.model(batch['src_seq'], batch['src_struct'], batch['seq_mask'], batch['struct_mask'], batch['trg_seq'], batch['length'], batch['gc_content'])
            else:
                logits = self.model(batch['src_struct'], batch['trg_seq'], batch['length'], gc_content=batch['gc_content'])            
            target = batch['trg_seq'][:, 1:]
            loss = self.loss_valid(logits.view(-1,logits.shape[-1]), target.contiguous().view(-1))
            loss = loss.view(target.shape)
            loss = loss.sum(1)/batch['length']
            perplexity = torch.exp(loss)
            metrics['epoch_perplexity'].extend(perplexity.detach().cpu())
            perplexity = perplexity.mean()
            loss = loss.mean()
            
            pred_seq = torch.argmax(logits, dim=-1)
            accuracy = torch.logical_and(pred_seq == target, target != self.ignore_index).sum(1) / batch['length']
            metrics['accuracy'].extend(accuracy)

            self.log(
                f"val/{self.val_sets_name[dataset_index]}/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=dataset_index == None or dataset_index == 0,
                sync_dist=True,
            )
            self.log(
                f"val/{self.val_sets_name[dataset_index]}/perplexity",
                perplexity,
                on_step=False,
                on_epoch=True,
                prog_bar=dataset_index == None or dataset_index == 0,
                sync_dist=True,
            )
        return_dict = {"loss": loss,
                       "batch_length": torch.mean(batch['length'].float()),
                       "count": batch['length'].shape[0]}

        metrics = {k: torch.stack(v).sum() for k, v in metrics.items()}
        return_dict.update(metrics)
        self.valid_outputs.append(return_dict)

        if self.val_fold_test and (self.final_val or ((self.current_epoch +1) % self.cfg_train.val_fold_test_freq == 0)):
            generated_seq = []
            for i in range(self.n_samples):
                if self.constrained_design:
                    batch_out = self.model.generate(batch['src_seq'], batch['src_struct'], batch['seq_mask'], batch['struct_mask'], batch['length'], batch['gc_content'], greedy=False).cpu().unsqueeze(1)
                else:
                    batch_out = self.model.generate(batch['src_struct'], batch['length'], greedy=False).cpu().unsqueeze(1)
                generated_seq.append(batch_out)
            generated_seq = torch.cat(generated_seq, dim=1)
            if self.cfg_train.devices > 1:
                #self.py_logger.info(generated_seq.shape)
                #print(generated_seq.shape)
                exp_seq = torch.zeros(self.cfg_train.batch_size, self.n_samples, self.cfg_train.max_len)
                exp_seq[:generated_seq.shape[0], :generated_seq.shape[1], :generated_seq.shape[2]] = generated_seq
                self.generated_seq.append(exp_seq)
                if self.final_val:
                    exp_batch = torch.zeros(self.cfg_train.batch_size, self.cfg_train.max_len, self.cfg_train.max_len)
                    exp_batch[:batch['src_struct'].shape[0], :batch['src_struct'].shape[1], :batch['src_struct'].shape[1]] = batch['src_struct']
                    self.val_lengths.append(batch['length'])
                    self.val_batches.append(exp_batch)
            else:
                for i in range(len(generated_seq)):
                    self.generated_seq.append(generated_seq[i][:, :batch['length'][i]])
        
        return return_dict
    
    def on_validation_batch_end_(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        outputs = [self.valid_outputs]
        for dataset_name, output in zip(self.val_sets_name, outputs):

            if len(output) < 1:
                continue

            values = ["accuracy", "batch_length", "count", "epoch_perplexity"]

            summed_values = {k: 0 for k in values}
            for out_dict in output:
                for key in values:
                    summed_values[key] += out_dict[key]

            metrics = {"batch_length": summed_values['batch_length'] / len(output),
                       "batch_size": summed_values['count'] / len(output)}

            for k in values:
                if k not in ["batch_length", "count"]:
                    metrics[k] = summed_values[k] / summed_values['count']

            for name, value in metrics.items():
                self.log(f"val/{dataset_name}/{name}", value,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, )
                if self.local_rank == 0:
                    print(f"val/{dataset_name}/{name}", value, self.local_rank)
        self.valid_outputs = []
        if self.val_fold_test and self.generated_seq != []:
            if self.cfg_train.devices > 1:
                self.generated_seq = torch.cat(self.generated_seq, dim=0)
            all_generated_seq = self.all_gather(self.generated_seq)
            if self.final_val and self.trainer.is_global_zero:
                if self.cfg_train.devices > 1:
                    val_batches = torch.cat(self.val_batches, dim=0)
                    all_val_batches = self.all_gather(val_batches)
                    val_lengths = torch.cat(self.val_lengths, dim=0)
                    all_val_lengths = self.all_gather(val_lengths)
                    torch.save(all_val_lengths, os.path.join(self.save_dir, f"val_lengths_final.pt"))
                    torch.save(all_val_batches, os.path.join(self.save_dir, f"val_batches_final.pt"))
                torch.save(all_generated_seq, os.path.join(self.save_dir, f"generated_seq_final.pt"))         
            elif self.trainer.is_global_zero:
                torch.save(all_generated_seq, os.path.join(self.save_dir, f"generated_seq_{self.current_epoch}.pt"))
            self.trainer.strategy.barrier() #to let other cards to wait
        self.generated_seq = []
        
            

    def predict_step(self, batch: Any, batch_idx: int):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch['src_struct'], batch['trg_seq'], batch['length'])
            pred_seq = torch.argmax(logits, dim=-1)
        return pred_seq
    
    def configure_optimizers(self):
        if 'optimizer_param_grouping' in self.cfg_train:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(self.model, self.cfg_train.optimizer,
                                                        **self.cfg_train.optimizer_param_grouping)
        else:
            parameters = self.model.parameters()
        optimizer = instantiate(self.cfg_train.optimizer, parameters)

        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            self.py_logger.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

        if 'scheduler' not in self.cfg_train:
            return optimizer
        else:
            lr_lambda = get_learning_rate_schedule(self.cfg_train.scheduler)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': self.cfg_train.get('scheduler_interval', 'step'),
                                 'monitor': self.cfg_train.get('scheduler_monitor', 'val/loss')}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        # TD [2022-04-30]: DeepSpeed optimizer uses the kwarg set_grad_to_none instead of set_to_none
        if 'set_to_none' in inspect.signature(optimizer.zero_grad).parameters:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()