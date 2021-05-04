
import glob
import numpy as np
import math
import re
import random as rd

def pad(original_array):
    # Divide into pMHC and TCR
    pmhc = np.concatenate((original_array[original_array[:, 20] == 1], original_array[original_array[:, 21] == 1]))
    tcr = np.concatenate((original_array[original_array[:, 22] == 1], original_array[original_array[:, 23] == 1]))
    # Padding pMHC (only at the end)
    padding_size = (192 - pmhc.shape[0])
    end_pad = np.zeros((padding_size, pmhc.shape[1]))
    pmhc_padded = np.concatenate((pmhc, end_pad))
    # Padding TCR
    padding_size = (228 - tcr.shape[0]) / 2
    front_pad = np.zeros((math.floor(padding_size), tcr.shape[1]))
    end_pad = np.zeros((math.ceil(padding_size), tcr.shape[1]))
    tcr_padded = np.concatenate((front_pad, tcr, end_pad))
    # Concatanate pMHC and TCR
    array_padded = np.concatenate((pmhc_padded, tcr_padded))
    return array_padded


# REMOVE HALF OF NEGATIVES FIRST

# 20 one-hot-AA
# 7 per_res_fa_atr, per_res_fa_rep, per_res_fa_sol, per_res_fa_elec, per_res_fa_dun, per_res_p_aa_pp, per_res_score
# 6 foldx_MP,	foldx_MA,	foldx_MB,	foldx_PA,	foldx_PB,	foldx_AB
# 4 global_complex_total_score, global_complex_fa_atr, global_complex_fa_dun, global_complex_fa_elec,
# 3 global_complex_fa_rep, global_complex_fa_sol, global_complex_p_aa_pp,
# 7 repeat tcr
# 7 repeat mhc

one_hot = "A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,"
per_res = "per_res_fa_atr, per_res_fa_rep, per_res_fa_sol, per_res_fa_elec, " \
          "per_res_fa_dun, per_res_p_aa_pp, per_res_score,"
foldx = "foldx_MP,foldx_MA,foldx_MB,foldx_PA,foldx_PB,foldx_AB,"
global_complex = "global_complex_total_score, global_complex_fa_atr, global_complex_fa_dun, global_complex_fa_elec," \
                "global_complex_fa_rep, global_complex_fa_sol, global_complex_p_aa_pp,"
global_tcr = "global_tcr_total_score, global_tcr_fa_atr, global_tcr_fa_dun, global_tcr_fa_elec," \
                "global_tcr_fa_rep, global_tcr_fa_sol, global_tcr_p_aa_pp,"
global_pmhc = "global_pmhc_total_score, global_pmhc_fa_atr, global_pmhc_fa_dun, global_pmhc_fa_elec," \
                "global_pmhc_fa_rep, global_pmhc_fa_sol, global_pmhc_p_aa_pp,"
header = one_hot + per_res + foldx + global_complex + global_tcr + global_pmhc

for partition in range(1,6):
    print(partition)
    pos_files_in_part = glob.glob(f"data/train_data/*{partition}p*_*pos*")
    tenx_files_in_part = glob.glob(f"data/train_data/*{partition}p*tenx*")
    swap_files_in_part = glob.glob(f"data/train_data/*{partition}p*swap*")
    tenx_chosen_files = rd.sample(tenx_files_in_part, len(swap_files_in_part)*2)
    filelist = pos_files_in_part + tenx_chosen_files + swap_files_in_part
    partition_arr = np.empty(shape = (len(filelist), 420, 54))
    partition_labels_arr = np.empty(shape = (len(filelist)))
    i = 0
    for file in filelist:
        r = re.search(r'pos', file)
        if r:
            partition_labels_arr[i] = 1
        else:
            partition_labels_arr[i] = 0
        arr = np.load(file)
        padded_arr = pad(arr)
        new_arr = np.concatenate((padded_arr[:, 0:20],
                                  padded_arr[:, 24:27], padded_arr[:, [30]], padded_arr[:, 38:40], padded_arr[:, [43]],
                                  padded_arr[:, 64:70],
                                  padded_arr[:, [71]], padded_arr[:, 74:77], padded_arr[:, 79:81], padded_arr[:, [89]],
                                  padded_arr[:, [95]], padded_arr[:, 98:101], padded_arr[:, 103:105],
                                  padded_arr[:, [113]],
                                  padded_arr[:, [119]], padded_arr[:, 122:125], padded_arr[:, 127:129],
                                  padded_arr[:, [137]]), axis=1)
        partition_arr[i,:,:] = new_arr
        i += 1
    filename = f"P{partition}_input.npz"
    np.savez(filename, partition_arr)
    filename = f"P{partition}_labels.npz"
    np.savez(filename, partition_labels_arr)

outfile = open("example.csv", "w")
outfile.write(header+"\n")
for row in new_arr:
    string = ""
    for element in row:
        string += str(element)
        string += ","
    string += "\n"
    outfile.write(string)
outfile.close

print(file)