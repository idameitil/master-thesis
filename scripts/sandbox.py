import glob
import numpy as np

files = glob.glob("./data/train_data/*")
for file in files:
    arr = np.load(file)
    new_arr = arr[:,:142]
    filename = file.replace("./data/train_data/", "")
    outpath = "./data/train_data_new/" + filename
    np.save(outpath, new_arr)