import numpy as np

#from fastai.basic_data import *
#from fastai.basic_train import *
#from fastai.callbacks import *
#from fastai.data_block import *
#from fastai.metrics import *
#from fastai.train import *
#from fastai.utils import *
#from fastai.core import *
#from fastai.gen_doc import *

import torch
import glob
import re

from sklearn.preprocessing import Normalizer

def upsample(X, y):
    Xp = []
    yp = []

    threshold = 0.5

    neg_index = np.where(y == 0)[0]
    pos_index = np.where(y == 1)[0]

    choices_neg = list(neg_index)
    choices_pos = list(pos_index)

    while len(choices_neg) > 0:
        chance = np.random.rand()

        if chance > threshold:
            choice = np.random.choice(choices_pos)
            choices_pos.remove(choice)
            if len(choices_pos) == 0:
                choices_pos = list(pos_index)

            y_value = 1

        if chance <= threshold:
            choice = np.random.choice(choices_neg)
            choices_neg.remove(choice)
            y_value = 0

        Xp.append(X[choice])
        yp.append(y_value)

    # Create numpy array
    dim1 = len(Xp)
    dim2 = Xp[0].shape[0]
    dim3 = Xp[0].shape[1]

    df = np.zeros(shape = (dim1,dim2,dim3))
    
    for i in range(0, dim1):
        df[i] = Xp[i]
        
    Xp = df
    yp = np.array(yp)
    print(Xp.shape, yp.shape)
    return(Xp, yp)
    
# Random projection module
def generate_weights(batch_size, n_hid, new, GPU):
    if new == 1:
        print("Creating new m1, m2")
        #Gaussian noise
        batch_size = batch_size
        n_hid = n_hid
        # Final maxpool layer default gives size 32 * 20 = 640. With additional maxpool = 16 * 20 = 320.
        bs = batch_size
        n_hid = int(n_hid)

        #Create pre-set gaussian noise vector
        w1 = np.random.normal(size = (int(1), int(n_hid))).astype(np.float32)
        w2 = np.random.normal(size = (int(1), int(n_hid))).astype(np.float32)

        transformer = Normalizer().fit(w1)
        w1 = transformer.transform(w1)
        w1 = np.vstack([w1] * int(bs/2))

        transformer = Normalizer().fit(w2)
        w2 = transformer.transform(w2)
        w2 = np.vstack([w2] * int(bs/2))
        print(w2.shape)

        if GPU:
            m1 = torch.from_numpy(np.append(w1,w2)).view(bs, n_hid).cuda()
        else:
            m1 = torch.from_numpy(np.append(w1, w2)).view(bs, n_hid)
        if GPU:
            m2 = torch.from_numpy(np.append(w2,w1)).view(bs, n_hid).cuda()
        else:
            m2 = torch.from_numpy(np.append(w2, w1)).view(bs, n_hid)

        np.save("../data/m1.npy", m1.cpu())
        np.save("../data/m2.npy", m2.cpu())

    elif glob.glob("../data/m1.npy") and glob.glob("../data/m2.npy"):
        print("Loading m1, m2")
        if GPU:
            m1 = torch.from_numpy(np.load("../data/m1.npy")).cuda()
        else:
            m1 = torch.from_numpy(np.load("../data/m1.npy"))
        if GPU:
            m2 = torch.from_numpy(np.load("../data/m2.npy")).cuda()
        else:
            m2 = torch.from_numpy(np.load("../data/m2.npy"))

    return m1, m2

### Start position 
def data_generator(train, valid, test):
    
    filelist = train + valid + test; len(filelist)
    print(len(train), len(valid), len(test))
    data_size = len(filelist)
    ix_val = len(train)
    ix_test = len(train) + len(valid)

    filelist_loaded = []

    #Load data into dfs
    for i in range(0, len(filelist)):
        df = np.flipud(np.load(filelist[i]))[0:388,:]
        filelist_loaded.append(df)

    #Initialize empty df ordered by complexes and aminos
    dim1 = range(0, data_size)
    dim2 = filelist_loaded[0].shape[0]
    dim3 = filelist_loaded[0].shape[1]
    x = np.zeros(shape = (data_size, dim2, dim3))

    for i in range(0, data_size):
        x[i] = filelist_loaded[i]

    #Extract y
    y = np.zeros(shape = (data_size), dtype="int64")

    counter_x = range(0, data_size)
    counter_y = range(len(y))
    for c_x, c_y in zip(counter_x, counter_y):
        r = re.compile(r'.*pos.*')
        if bool(r.match(filelist[c_x])):
            y[c_y] = 1

    X_train, y_train = x[0 : ix_val], y[0 : ix_val]
    X_val, y_val = x[ix_val : ix_test], y[ix_val : ix_test]
    X_test, y_test = x[ix_test : ], y[ix_test : ]
    return X_train, y_train, X_val, y_val, X_test, y_test
