#!/bin/bash

for ds in syn_ns syn_hk syn_pdb riboswitch
do 
    for v in 0 1 2
    do
        python eval.py --path runs --v $v --ds $ds
        python eval.py --path runs --v $v --ds $ds --gc True
    done
done

python comb_metrics.py

python eval_antarna.py
python eval.py --path runs/learna_suite/learna/ --ds learna
python eval.py --path runs/learna_suite/liblearna/ --ds learna
python eval.py --path runs/learna_suite/liblearna_gc/ --ds learna --gc True
python eval.py --path runs/learna_suite/meta_learna/ --ds learna
python eval.py --path runs/learna_suite/meta_learna_adapt/ --ds learna
python eval.py --path runs/samfeo/ --ds samfeo