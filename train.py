import os
import argparse, collections, yaml
import logging
import torch.cuda
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np

from RNAinformer.pl_modules.rna_datamodule import DataModuleRNA
from RNAinformer.pl_modules.ribo_datamodule import DataModuleRibo
from RNAinformer.pl_modules.rna_trainer import RNADesignTrainer
from RNAinformer.utils.configuration import Config
from RNAinformer.utils.data.rna import CollatorMaskDesignMat, CollatorMaskGCDesignMat, CollatorRNADesignMatGC



def bold(msg):
    return f"\033[1m{msg}\033[0m"


def main(cfg):
    """
    Launch pretraining
    """
    torch.set_float32_matmul_precision('medium')
    if os.environ.get("LOCAL_RANK") is None or os.environ.get("LOCAL_RANK") == 0:
        is_rank_zero = True
        rank = 0
    else:
        is_rank_zero = False
        rank = os.environ.get("LOCAL_RANK")

    if isinstance(cfg.trainer.devices, str):
        cfg.trainer.devices = list(map(int, cfg.trainer.devices.split(",")))
        cfg.rna_data.num_gpu_worker = len(cfg.trainer.devices)

    logger = logging.getLogger(__name__)

    if is_rank_zero:
        logging.basicConfig(
            format="[%(asctime)s][%(levelname)s][%(name)s] - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(torch.cuda.get_device_name())
        logger.info(bold("######################################################"))
        logger.info(bold("########          START   TRAINING          ##########"))
        logger.info(bold("######################################################"))

        logger.info(bold("############### CONFIGURATION"))
        logger.info("RNA Task args")
        logger.info(cfg.rna_data)
        logger.info("Trainer args")
        logger.info(cfg.trainer)
        logger.info("Train args")
        logger.info(cfg.train)
        logger.info("Optimizer args")
        logger.info(cfg.train.optimizer)
        logger.info("RNADesignFormer args")
        logger.info(cfg.RNADesignFormer)

    # Set seed before initializing model
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)

    logger.info(bold(f"############### LOAD DATA on rank {rank}"))
    if cfg.constrained_design:
        data_module = DataModuleRibo(**cfg.pd_train_data, logger=logger)
    else:
        data_module = DataModuleRNA(**cfg.rna_data, logger=logger)
    data_module.prepare_data()
    
    cfg.RNADesignFormer.src_vocab_size = data_module.struct_vocab_size
    cfg.RNADesignFormer.trg_vocab_size = data_module.seq_vocab_size
    cfg.RNADesignFormer.struct_vocab_size = data_module.struct_vocab_size
    cfg.RNADesignFormer.seq_vocab_size = data_module.seq_vocab_size
    logger.info(f'#### Load logger on rank {rank}')
    
    training_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="runs",
        name=cfg.experiment.session_name,
        prefix="",
    )
    save_dir = training_logger.log_dir
    model_module = RNADesignTrainer(
        cfg_train=cfg.train,
        cfg_model=cfg.RNADesignFormer,
        py_logger=logger,
        val_sets_name=data_module.valid_sets,
        ignore_index=data_module.ignore_index,
        save_dir=save_dir,
        constrained_design=cfg.constrained_design,
    )

    if is_rank_zero:
        def count_parameters(parameters):
            return sum(p.numel() for p in parameters if p.requires_grad)

        logger.info(f"#### trainable_parameters {count_parameters(model_module.parameters())}")

        def print_model_param_stats(model):
            for idx, (name, params) in enumerate(model.named_parameters()):
                logger.info(
                    f"{idx:03d} {name:70} shape:{str(list(params.shape)):12} mean:{params.mean():8.4f} std:{params.std():8.6f} grad: {params.requires_grad}")

        print_model_param_stats(model_module.model)

    if cfg.resume_training:
        logger.info(bold(f"############### RESUME TRAINING on rank {rank}"))

    logger.info(f'#### Load logger on rank {rank}')
    if cfg.trainer.devices > 1:
        strategy = "ddp" 
        if cfg.RNADesignFormer.get('seq_mask', False):
            strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"
    cfg.save_config(training_logger.log_dir, file_name="config.yaml")
    trainer = pl.Trainer(max_steps=cfg.trainer.max_steps,
                      accelerator="gpu",
                      strategy=strategy,
                      logger=training_logger,
                      log_every_n_steps=1,
                      callbacks=[LearningRateMonitor(logging_interval='step')],
                      enable_progress_bar=True,
                      enable_checkpointing=True,
                      devices=cfg.trainer.devices,
                      num_nodes=cfg.trainer.num_nodes,
                      precision=cfg.trainer.precision,
                      #accumulate_grad_batches=4,
                    )
    
    logger.info(bold(f"############### TRAINER on rank {rank}"))
     # uses multiple GPUs but all on 1 instance


    logger.info(f"Starting training on rank {rank}")

    trainer.fit(model=model_module, datamodule=data_module)
    val_dataloader = data_module.val_dataloader()
    model_module.final_val = True
    trainer = pl.Trainer(accelerator="gpu",devices=1, num_nodes=1,logger=training_logger,precision=cfg.trainer.precision)
    trainer.validate(model=model_module, dataloaders=val_dataloader)
    
   


if __name__ == "__main__":

    from functools import reduce  # forward compatibility for Python 3
    import operator


    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d


    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)


    def setInDict(dataDict, mapList, value):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


    def convert_string_value(value):
        if value in ('false', 'False'):
            value = False
        elif value in ('true', 'True'):
            value = True
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
        return value


    default_config_name = "default_config.yaml"

    parser = argparse.ArgumentParser(description='Train RNADesignformer')
    parser.add_argument('-c', '--config', type=str, default=default_config_name, help='config file name')
    parser.add_argument('-d', '--dataset', type=str, default='rfam')#,choices=['rfam','bprna','pdb','riboswitch','rna3d',"syn_ns"], help='experiment dataset name')
    parser.add_argument('-gc', '--gc', type=bool, default=False, help='GC content conditioning')

    args, unknown_args = parser.parse_known_args()

    config_name = args.config
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'

    config_file = os.path.join("config", args.config)
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)

    for arg in unknown_args:
        if '=' in arg:
            keys = arg.split('=')[0].split('.')
            value = convert_string_value(arg.split('=')[1])
            print(keys, value)
            setInDict(config_dict, keys, value)
        else:
            raise UserWarning(f"argument unknown: {arg}")

    if args.dataset != 'rfam':
        for k in config_dict[f"{args.dataset}_data"]:
            if k in config_dict['rna_data']:
                config_dict['rna_data'][k] = config_dict[f"{args.dataset}_data"][k]
        for k in config_dict[f"{args.dataset}_data"]['test']:
            config_dict['test'][k]=config_dict[f"{args.dataset}_data"]['test'][k]
    config_dict['constrained_design'] = args.dataset == 'riboswitch'
    config = Config(config_dict=config_dict)    
    main(cfg=config)