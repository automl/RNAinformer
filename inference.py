import logging
import torch
import os, sys, socket
import argparse, collections, yaml
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from RNAinformer.pl_modules.rna_datamodule import IGNORE_INDEX, PAD_INDEX
from RNAinformer.utils.configuration import Config
from RNAinformer.model.RNADesignFormer import RNADesignFormer
from RNAinformer.model.RiboDesignFormer import RiboDesignFormer
from RNAinformer.utils.data.rna import CollatorRNADesignMat, CollatorRiboDesignMat
from torch.utils.data import SequentialSampler,BatchSampler

def infer_ribo(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    collator =CollatorRiboDesignMat(PAD_INDEX, IGNORE_INDEX)
    test_data = torch.load("data/riboswitch/ribo_design_all_len2100_designTrue_seed1_v3.pth")['train']
    gc_bands = torch.load("data/riboswitch/gc_bands_ribo.pt")  
    batch_size = cfg.test.batch_size
    num_samples = cfg.test.n_samples
    model = RiboDesignFormer(cfg.RNADesignFormer)
    state_dict = torch.load(cfg.model_path,map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model." in k}
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    output_path = cfg.path + f'/predictions_{num_samples}/'
    if cfg.RNADesignFormer.get('flash', False):
        model.to(torch.float16)
    if not cfg.gc:
        os.makedirs(output_path, exist_ok=True)
        outputs=[]
        sampler = BatchSampler(SequentialSampler(torch.arange(len(test_data))), batch_size, drop_last=False)
        for i in tqdm(list(sampler)):
            batch = collator([test_data[j] for j in i])
            batch_out=model.generate(batch['src_seq'].cuda(), batch['src_struct'].cuda(), batch['seq_mask'].cuda(), batch['struct_mask'].cuda(), batch['length'].cuda(),greedy=True,constrained_generation=cfg.constrained_generation).cpu().unsqueeze(1)
            for _ in range(num_samples-1):
                out = model.generate(batch['src_seq'].cuda(), batch['src_struct'].cuda(), batch['seq_mask'].cuda(), batch['struct_mask'].cuda(), batch['length'].cuda(),greedy=False,constrained_generation=cfg.constrained_generation).cpu().unsqueeze(1)
                batch_out = torch.cat([batch_out, out], dim=1)
            for i in range(batch_out.shape[0]):
                outputs.append(batch_out[i][:, :batch['length'][i]])
        torch.save(outputs, output_path+"ribo_outputs.pt")
    else:
        outputs=defaultdict(list)
        gc_targets = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        filtered_gc = True
        output_path += "gc/"
        os.makedirs(output_path, exist_ok=True)
        for gc in gc_targets:
            if filtered_gc:
                data = [test_data[i] for i in gc_bands[gc]]
                sampler = BatchSampler(SequentialSampler(torch.arange(len(data))), batch_size, drop_last=False)
            else:
                data = test_data
            for i in tqdm(list(sampler)):
                batch = collator([data[j] for j in i])
                batch['gc_content'] = torch.ones_like(batch['gc_content'],dtype=torch.float32)*gc
                if cfg.RNADesignFormer.get('flash', False):
                    batch['gc_content'] = batch['gc_content'].to(torch.float16)
                batch_out=model.generate(batch['src_seq'].cuda(), batch['src_struct'].cuda(), batch['seq_mask'].cuda(), batch['struct_mask'].cuda(), batch['length'].cuda(),batch['gc_content'].cuda(),greedy=True,constrained_generation=cfg.constrained_generation).cpu().unsqueeze(1)
                for _ in range(num_samples-1):
                    out = model.generate(batch['src_seq'].cuda(), batch['src_struct'].cuda(), batch['seq_mask'].cuda(), batch['struct_mask'].cuda(), batch['length'].cuda(),batch['gc_content'].cuda(),greedy=False,constrained_generation=cfg.constrained_generation).cpu().unsqueeze(1)
                    batch_out = torch.cat([batch_out, out], dim=1)
                for i in range(batch_out.shape[0]):
                    outputs[gc].append(batch_out[i][:, :batch['length'][i]])
        torch.save(outputs, output_path+"ribo_outputs_gc.pt")
        torch.save(gc_targets, output_path+"gc_targets.pt")

def infer(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    collator = CollatorRNADesignMat(PAD_INDEX, IGNORE_INDEX)
    batch_size = cfg.test.batch_size
    num_samples = cfg.test.n_samples
    model = RNADesignFormer(cfg.RNADesignFormer)
    state_dict = torch.load(cfg.model_path,map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model." in k}
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    output_path = cfg.path + f'/predictions/'
    if cfg.gc:
        output_path += "gc/"
    elif cfg.energy:
        output_path += "energy/"
        
    os.makedirs(output_path, exist_ok=True)
    for i,ds in enumerate(cfg.test.datasets[:]):
        print("Runing inference on dataset: ", ds)
        test_data = torch.load(f'{cfg.test.cache_dir}/{ds}.pt')
        test_data = [i for i in test_data if i['length']<= cfg.rna_data.max_len]
        sampler = BatchSampler(SequentialSampler(torch.arange(len(test_data))), batch_size, drop_last=False)
        outputs=[]
        for i in tqdm(list(sampler)):
            batch = collator([test_data[j] for j in i])
            if not cfg.gc:
                gc = None
            else:
                gc = batch['gc_content'].cuda()
            if not cfg.energy:
                energy = None
            else:
                energy = batch['energy'].cuda()
            if cfg.RNADesignFormer.get('flash', False):
                model.to(torch.float16)
                if cfg.gc:
                    gc = gc.to(torch.float16)
                if cfg.energy:
                    energy = energy.to(torch.float16)
            batch_out=model.generate(batch['src_struct'].cuda(), batch['length'].cuda(),None,gc,energy,greedy=True).cpu().unsqueeze(1)
            for _ in range(num_samples-1):
                out = model.generate(batch['src_struct'].cuda(), batch['length'].cuda(),None,gc,energy,greedy=cfg.greedy).cpu().unsqueeze(1)
                batch_out = torch.cat([batch_out, out], dim=1)
            outputs.extend(batch_out)
        final_outputs = []
        for i in range(len(outputs)):
            final_outputs.append(outputs[i][:,:test_data[i]['length']])
        torch.save(final_outputs, output_path+f'{ds}_preds.pt')
    cfg.save_config(output_path,file_name="config.yaml")

if __name__ == '__main__':
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--path', type=str)
    parser.add_argument('--gc', type=bool, default=False)
    parser.add_argument('--energy', type=bool, default=False)
    parser.add_argument('--greedy', type=bool, default=False)
    parser.add_argument('--constrained_generation',type=bool, default=False)
    parser.add_argument('--flash', type=bool, default=True)

    args, unknown_args = parser.parse_known_args()    
    config_file = os.path.join(args.path,'config.yaml')
    model_path = glob(os.path.join(args.path,'checkpoints','*.ckpt'))[0]
    if len(glob(os.path.join(args.path,'checkpoints','*.state')))>0:
        model_path = glob(os.path.join(args.path,'checkpoints','*.state'))[0]
    
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    for i in vars(args):
        if vars(args)[i] is not None:
            config_dict[i] = vars(args)[i]
    
    for arg in unknown_args:
        if '=' in arg:
            keys = arg.split('=')[0].split('.')
            value = convert_string_value(arg.split('=')[1])
            print(keys, value)
            setInDict(config_dict, keys, value)
        else:
            raise UserWarning(f"argument unknown: {arg}")
   
    config_dict['model_path'] = model_path
    config_dict['config_path'] = config_file
    if not args.flash:
        config_dict['RNADesignFormer']['flash'] = False
    cfg = Config(config_dict=config_dict)
    with torch.no_grad():
        if args.constrained_generation :
            infer_ribo(cfg)
        else:
            infer(cfg)

