import pandas as pd
seq = {"aptamer":"AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA","spacer":"?NNNNNN?","comp_region":"UGAAGUGCUG?", "8U":"UUUUUUUU"}
struct = {"aptamer":"........NNN(((((.....)))))...NNN((((((((((", "spacer":"?NN....?", "comp_region":"))))))))))?", "8U":"N......."}
lengths = {"spacer": 20, "comp_region": 21}
comp_region = - len(seq["comp_region"]) + 1 + lengths["comp_region"]
s=[]
for i in range(15):
    for j in range(15-i):
        if j + i <= 14:
            s.append((i,j))
samples = []
u_seq = []
u_struct = []
for i,j in s:
    for k in range(12):
        sq = seq["aptamer"] + "N"*i + seq["spacer"][1:-1] + "N"*j + seq["comp_region"][:-1] + "N"*k + seq["8U"]
        st = struct["aptamer"] + "N"*i + struct["spacer"][1:-1] + "N"*j + struct["comp_region"][:-1] + "N"*k + struct["8U"]
        if sq not in u_seq or st not in u_struct:
            u_seq.append(sq)
            u_struct.append(st)
            assert len(sq) == len(st)
            samples.append({"target_sequence":sq, "target_structure":st})
print(len(samples))
for i in samples:
    if 91 < len(i['target_sequence']) < 61:
        print(i)
samples = pd.DataFrame(samples)
samples.to_pickle("RNA-design/data/riboswitch/ribo_design_all.plk")


