import numpy as np

arr = np.load(file = "./data/train_data_new/2294_2p_neg_tenx.npy")

per_res_names = ["fa_atr","fa_rep","fa_sol","fa_intra_rep","fa_intra_sol_xover4",
                 "lk_ball_wtd","fa_elec","pro_close","hbond_sr_bb","hbond_lr_bb",
                 "hbond_bb_sc","hbond_sc","dslf_fa13","omega","fa_dun","p_aa_pp",
                 "yhh_planarity","ref","rama_prepro", "score"]

total_names = ["total_score","score","dslf_fa13","fa_atr","fa_dun","fa_elec",
               "fa_intra_rep","fa_intra_sol_xover4","fa_rep","fa_sol",
               "hbond_bb_sc","hbond_lr_bb","hbond_sc","hbond_sr_bb","linear_chainbreak",
               "lk_ball_wtd","omega","overlap_chainbreak","p_aa_pp","pro_close","rama_prepro",
               "ref","time","yhh_planarity"]

per_res_complex = ""
per_res_separate = ""
for name in per_res_names:
    per_res_complex += (f"per_res_complex_{name},")
    per_res_separate += (f"per_res_separate_{name},")

total_complex = ""
totaL_tcr = ""
total_pmhc = ""
for name in total_names:
    total_complex += (f"total_complex_{name},")
    totaL_tcr += (f"total_tcr_{name},")
    total_pmhc += (f"total_pmhc_{name},")

one_hot = "A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,"
chain = "MHC,peptide,TCRA,TCRB,"
foldx = "foldx_MP,foldx_MA,foldx_MB,foldx_PA,foldx_PB,foldx_AB,"

header = one_hot + chain + per_res_complex + per_res_separate + foldx + \
         total_complex + totaL_tcr + total_pmhc + "\n"

outfile = open("../example_array.csv", "w")
outfile.write(header)
for row in arr:
    string = ""
    for element in row:
        string += str(element)
        string += ","
    string += "\n"
    outfile.write(string)
outfile.close