import glob
import pandas as pd

def comb_metrics(path,n,gc,metrics):
    if n is not None:
        if n == 20:
            g_path = path + f"*/predictions_20/"
        else:
            g_path = path + f"*/predictions/"
    else:
        g_path = path + f"*/"
    save_path = path + f"metrics.csv"
    if gc:
        g_path = g_path + "gc/"
        save_path = path + f"metrics.csv"
    files = glob.glob(g_path + f"{metrics}.csv")
    cdfs = []
    for i in files:
        version = i.split("/")[2]
        df = pd.read_csv(i)
        df["Version"] = version
        print(i)
        cdfs.append(df)
    cdf = pd.concat(cdfs)
    cdf.fillna(0,inplace=True)
    if 'ribo' in path and not gc:
        m = cdf.mean(numeric_only=True)
        s = cdf.std(numeric_only=True)
        cdf.loc[f"Mean"] = round(m,3)
        cdf.loc[f"Std"] = round(s,3)
    elif 'ribo' in path and gc:
        for test_gc in cdf['gc'].unique():
            m = cdf[cdf['gc'] == test_gc].mean(numeric_only=True)
            s = cdf[cdf['gc'] == test_gc].std(numeric_only=True)
            cdf.loc[f"{test_gc} Mean"] = round(m,3)
            cdf.loc[f"{test_gc} Std"] = round(s,3)
            cdf.loc[f"{test_gc} Mean","gc"] = test_gc
            cdf.loc[f"{test_gc} Std","gc"] = test_gc
    else:
        for test_set in cdf['Test_set'].unique():
            m = cdf[cdf['Test_set'] == test_set].mean(numeric_only=True)
            s = cdf[cdf['Test_set'] == test_set].std(numeric_only=True)
            cdf.loc[f"{test_set} Mean"] = round(m,3)
            cdf.loc[f"{test_set} Std"] = round(s,3)
            cdf.loc[f"{test_set} Mean","Test_set"] = test_set
            cdf.loc[f"{test_set} Std","Test_set"] = test_set
    cdf.to_csv(save_path)

comb_metrics("runs/syn_ns/",True,False,"metrics")
comb_metrics("runs/syn_ns_gc/",True,True,"metrics")

comb_metrics("runs/syn_hk/",True,False,"metrics")
comb_metrics("runs/syn_hk_gc/",True,True,"metrics")

comb_metrics("runs/syn_pdb/",True,False,"metrics")
comb_metrics("runs/syn_pdb_gc/",True,True,"metrics")

comb_metrics("runs/ribo/",20,False,"metrics")
comb_metrics("runs/ribo_gc/",20,True,"metrics")


