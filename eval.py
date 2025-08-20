import torch
import sys
from RNAinformer.utils.test_pk_mult import test_fold_pk_mult
from RNAinformer.utils.fold import test_fold
from glob import glob
import argparse
from collections import defaultdict
import pandas as pd
import pickle
from RNAinformer.utils.eval_ribo import eval_ribo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--v', type=int,default=None)
    parser.add_argument('--ds', type=str)
    parser.add_argument('--gc', type=bool, default=False)
    parser.add_argument('--max_len', type=int, default=200)

    args = parser.parse_args()
    if args.v is not None:
        if args.gc:
            args.path = f"runs/{args.ds}_gc/version_{args.v}/predictions/gc/"
        else:
            args.path = f"runs/{args.ds}/version_{args.v}/predictions/"
    if args.ds == 'riboswitch':
        if args.gc:
            args.path = f"runs/ribo_gc/version_{args.v}/predictions_20/gc/"
        else:
            args.path = f"runs/ribo/version_{args.v}/predictions_20/"
    model = args.path.split('/')[1:3]
    print(f'Evaluating model {model} on {args.ds} dataset')
    sys.stdout = open(args.path+f"test_results.txt", "w")
    metrics = defaultdict(list)
    if args.ds == 'rfam' or args.ds == 'syn_ns':
        for i in glob(f"{args.path}/*preds.plk.gz"):
            preds=pd.read_pickle(i, compression='tar')
            ds = i.split("/")[-1].replace("_preds.plk.gz","")
            test_data = torch.load(f'data/{args.ds}/{ds}.pt')
            test_data = [i for i in test_data if i['length']<= args.max_len]
            preds = [preds[preds['id']==i]['sequence'].values.tolist() for i in range(len(test_data))]
            preds = [i for i in preds if len(i[0])<= args.max_len]
            print("*"*20)
            print(f"Testing on Dataset: {ds}")
            metrics["Test_set"].append(ds)
            ds_metrics = test_fold(preds, test_data, args.max_len, args.gc)
            for k in ds_metrics.keys():
                metrics[k].append(ds_metrics[k])
            print("*"*20)
    elif args.ds == 'learna' or args.ds == 'samfeo':
        ds = "syn_ns"
        test_data = torch.load(f'data/{ds}/{ds}_test_learna.pt')
        test_data = [i for i in test_data if i['length']<= args.max_len]
        preds=pd.read_pickle(f"{args.path}/preds.plk.gz",compression='tar')
        preds = [preds[preds['id']==i]['sequence'].values.tolist() for i in range(len(test_data))] 
        preds = [i for i in preds if len(i[0])<= args.max_len]
        print("*"*20)
        print(f"Testing on Dataset: {ds}")
        metrics["Test_set"].append(ds)
        ds_metrics = test_fold(preds, test_data, args.max_len, args.gc)
        for k in ds_metrics.keys():
                metrics[k].append(ds_metrics[k])
        print("*"*20)
    elif args.ds == 'riboswitch':
        if args.gc:
            preds = pd.read_pickle(args.path+"/ribo_outputs_gc.plk.gz",compression='tar')
            gc_targets = preds['gc'].unique()
            for gc in gc_targets:
                print(f"Target GC content: {gc}")
                t_preds = preds[preds['gc']==gc]
                t_preds = [t_preds[t_preds['id']==i]['sequence'].values.tolist() for i in range(int(len(t_preds)/20))]
                gc_metrics = eval_ribo(t_preds, args.path, gc)
                metrics['gc'].append(gc)
                for k in gc_metrics.keys():
                    metrics[k].append(gc_metrics[k])
                print("*****************************")
            df=pd.DataFrame(metrics)
            df.to_csv(args.path+"/metrics.csv")
        else:
            preds = preds = pd.read_pickle(args.path+"/ribo_outputs.plk.gz",compression='tar')
            preds = [preds[preds['id']==i]['sequence'].values.tolist() for i in range(int(len(preds)/20))]
            metrics=eval_ribo(preds, args.path)
            df=pd.DataFrame(metrics,index=[0])
            df.to_csv(args.path+"/metrics.csv")
            print("*****************************")
    else:
        for i in glob(f"{args.path}/*structs.plk.gz"):
            preds=pd.read_pickle(i, compression='tar')            
            ds = i.split("/")[-1].replace("_preds_structs.plk.gz","")
            test_data = torch.load(f'data/{args.ds}/{ds}.pt')
            if 'pdb' in ds:
                seqs = [preds[preds['id']==i]['sequence'].values.tolist() for i in range(len(test_data))]
                preds = [preds[preds['id']==i]['structure'].values.tolist() for i in range(len(test_data))]
                test_data = [i for i in test_data if i['length']<= args.max_len]
            else:
                test_data = [i for i in test_data if i['length']<= args.max_len]
                seqs = [preds[preds['id']==i]['sequence'].values.tolist() for i in range(len(test_data))]
                preds = [preds[preds['id']==i]['structure'].values.tolist() for i in range(len(test_data))]
            preds = [preds[i] for i in range(len(preds)) if len(seqs[i][0])<= args.max_len]
            seqs = [i for i in seqs if len(i[0])<= args.max_len]
            print("*"*20)
            print(f"Testing on Dataset: {ds}")
            metrics["Test_set"].append(ds)
            ds_metrics = test_fold_pk_mult(preds, seqs, test_data, ds, args.gc)
            for k in ds_metrics.keys():
                metrics[k].append(ds_metrics[k])
            print("*"*20)
    df=pd.DataFrame(metrics)
    df.to_csv(args.path+"/metrics.csv")
    sys.stdout.close()