import glob
import numpy as np
import re

files = glob.glob("/home/projects/ht3_aim/people/idamei/data/train_data/*")
for file in files:
    arr = np.load(file)
    new_arr = arr[:,:142]
    filename = file.replace("/home/projects/ht3_aim/people/idamei/data/", "")
    outpath = "/home/projects/ht3_aim/people/idamei/data/train_data_new/" + filename
    np.save(outpath, new_arr)
