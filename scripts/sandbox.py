import sys
import numpy as np
data = np.load(file="/home/ida/TCR-p-MHC/data/03_Dataset/2ypl_3p_S1_1t_99s_pMHC-TCR.npy")
np.set_printoptions(threshold=sys.maxsize)
print(data)