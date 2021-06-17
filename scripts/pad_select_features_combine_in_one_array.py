import sys
import glob
import numpy as np
import math
import re
import random as rd
import pickle

def pad(original_array):
    # Divide into pMHC and TCR
    pmhc = np.concatenate((original_array[original_array[:, 20] == 1], original_array[original_array[:, 21] == 1]))
    tcr = np.concatenate((original_array[original_array[:, 22] == 1], original_array[original_array[:, 23] == 1]))
    mhc = original_array[original_array[:, 20] == 1]
    # Padding TCR
    padding_size = (228 - tcr.shape[0]) / 2
    front_pad = np.zeros((math.floor(padding_size), tcr.shape[1]))
    end_pad = np.zeros((math.ceil(padding_size), tcr.shape[1]))
    tcr_padded = np.concatenate((front_pad, tcr, end_pad))
    # Concatanate pMHC and TCR
    array_padded = np.concatenate((pmhc, tcr_padded))
    return array_padded

# Make dict of ID and peptide
infile = open("/home/projects/ht3_aim/people/idamei/data/all_data_final_origin.csv", "r")
peptides = ['GILGFVFTL', 'GLCTLVAML', 'NLVPMVATV', 'CLGGLLTMV', 'FLYALALLL', 'ILKEPVHGV', 'IMDQVPFSV', 'KLQCVDLHV', 'KTWGQYWQV', 'KVAELVHFL', 'KVLEYVIKV', 'LLFGYPVYV', 'MLDLQPETT', 'RMFPNAPYL', 'RTLNAWVKV', 'SLFNTVATL', 'SLLMWITQV', 'YLLEMLWRL']
id_peptide_dict = {}
for line in infile:
    if line.startswith("#"):
        continue
    ID, _, _, peptide, _, _, _ = line.split(",")
    peptide_index = peptides.index(peptide)
    id_peptide_dict[ID] = peptide_index
print(id_peptide_dict)
#{1:"GILGFVFTL", 2:"GLCTLVAML", 3:"NLVPMVATV", 4:"CLGGLLTMV", 5:"FLYALALLL", 6:"ILKEPVHGV", 7:"IMDQVPFSV", 8:"KLQCVDLHV", 9:"KTWGQYWQV", 10:"KVAELVHFL", 11:"KVLEYVIKV", 12:"LLFGYPVYV", 13:"MLDLQPETT", 14:"RMFPNAPYL", 15:"RTLNAWVKV", 16:"SLFNTVATL", 17:"SLLMWITQV", 18:"YLLEMLWRL"}

for partition in range(1,6):
    print(partition)
    pos_files_in_part = glob.glob(f"data/train_data/*{partition}p*_*pos*")
    tenx_files_in_part = glob.glob(f"data/train_data/*{partition}p*tenx*")
    swap_files_in_part = glob.glob(f"data/train_data/*{partition}p*swap*")
    filelist = pos_files_in_part + tenx_files_in_part + swap_files_in_part
    partition_arr = np.empty(shape = (len(filelist), 416, 54))
    partition_labels_arr = np.empty(shape = (len(filelist)))
    partition_origins_arr = np.empty(shape = (len(filelist)))
    partition_peptide_arr = np.empty(shape = (len(filelist)))
    i = 0
    for filename in filelist:
        # Get peptide
        ID = filename.replace("data/train_data/", "").split("_")[0]
        peptide_index = id_peptide_dict[ID]
        partition_peptide_arr[i] = peptide_index
        # Get origin
        r = re.search(r'pos', filename)
        if r:
            partition_labels_arr[i] = 1
            partition_origins_arr[i] = 0
        else:
            partition_labels_arr[i] = 0
            r = re.search(r'swap', filename)
            if r:
                partition_origins_arr[i] = 1
            else:
                partition_origins_arr[i] = 2
        # Get data array
        arr = np.load(filename)
        padded_arr, old_mhc, flag = pad(arr, flag, old_mhc)
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
    filename = f"P{partition}_origin.npz"
    np.savez(filename, partition_origins_arr)
    filename = f"P{partition}_peptides.npz"
    np.savez(filename, partition_peptide_arr)

sys.exit()

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
