import sys
import numpy as np

output_array = np.load(file="/home/ida/master-thesis/energy_output_arrays.npz")
np.set_printoptions(precision=2, threshold=sys.maxsize)
#print(output_array["5isz.fsa_model_TCR-pMHC"][400][0:20])
print(output_array["5isz.fsa_model_TCR-pMHC"][400][44:64])
