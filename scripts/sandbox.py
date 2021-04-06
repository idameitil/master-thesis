import numpy as np
import sys

my_arr = np.load(file = "/home/ida/master-thesis/data/train_data/10003_2p_neg_swap.npy")
np.set_printoptions(threshold=sys.maxsize)
#print(my_arr[:,20] == 1) #MHC
print(my_arr[my_arr[:,20] == 1, :])
#print(my_arr[:,21] == 1) #peptide
#print(my_arr[:,22] == 1 or my_arr[:,23] == 1) #tcr
