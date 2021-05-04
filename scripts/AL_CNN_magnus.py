#!/usr/bin/env python
# coding: utf-8

# Imports

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import math
import re
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix

import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
#from torch.utils.data import (
#    DataLoader,
#)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

def pad(filename):
    # Load array
    original_array = np.load(filename)
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

def load_data(filelist):
    padded_length = 192 + 228
    X = np.zeros(shape=(len(filelist), padded_length, 142))
    y = np.zeros(shape=len(filelist))
    for i in range(len(filelist)):
        filename = filelist[i]
        final_array = pad(filename)
        X[i] = final_array
        r = re.search(r'pos', filename)
        if r:
            y[i] = 1
        else:
            y[i] = 0
    return X, y

###############################
###    Load data            ###
###############################

print("loading data...")
p1_filelist = glob.glob("data/train_data/*1p*")[:100]
#p1_filelist_file = open("p1_filelist.txt")
#p1_filelist = []
#for line in p1_filelist_file:
#    p1_filelist.append(line.strip().replace("/Volumes/Work/TCR-pMHC/", "data/train_data/"))
X_train, y_train = load_data(p1_filelist)
nsamples, nx, ny = X_train.shape
print(nsamples,nx,ny)

p2_filelist = glob.glob("data/train_data/*2p*")[:50]
#p2_filelist_file = open("p2_filelist.txt")
#p2_filelist = []
#for line in p2_filelist_file:
#    p2_filelist.append(line.strip().replace("/Volumes/Work/TCR-pMHC/", "data/train_data/"))
X_test, y_test = load_data(p2_filelist)
nsamples, nx, ny = X_test.shape
print(nsamples,nx,ny)

print("Percent positive samples in train:")
print(len(y_train[y_train == 1])/len(y_train)*100)
print("Percent positive samples in test:")
print(len(y_test[y_test == 1])/len(y_test)*100)

# make the data set into one dataset that can go into dataloader
train_ds = []
for i in range(len(X_train)):
    train_ds.append([X_train[i], y_train[i]])

test_ds = []
for i in range(len(X_test)):
    test_ds.append([X_test[i], y_test[i]])

bat_size = 32
train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)

bat_size = 32
test_ldr = torch.utils.data.DataLoader(test_ds,batch_size=bat_size, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#train_ldr, test_ldr = train_ldr.to(device), test_ldr.to(devide)

###############################
###    Define network       ###
###############################

print("Initializing network")

# Hyperparameters
input_size = 420
num_classes = 1
learning_rate = 0.01
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Batchnorm0
        self.a_norm_0 = nn.BatchNorm1d(in_channel)

        #Pad
        #pad_size_pmhc = (512 - 192) / 2
        self.m_pad = nn.ConstantPad1d((160), 0)
        #pad_size_tcr = (512 - 228) / 2
        self.t_pad = nn.ConstantPad1d((142), 0)
        #self.p_pad = nn.ConstantPad1d((112, 113), 0)

        #Conv1
        self.m_conv_1 = nn.Conv1d(in_channels = in_channel, out_channels = n_hid, kernel_size = ks1, padding = pad1)
        self.m_norm_1 = nn.BatchNorm1d(n_hid)
        self.m_max_1 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.m_drop_1 = nn.Dropout(p = drop_prob)
        self.m_ReLU_1 = nn.ReLU()

        self.t_conv_1 = nn.Conv1d(in_channels = in_channel, out_channels = n_hid, kernel_size = ks1, padding = pad1)
        self.t_norm_1 = nn.BatchNorm1d(n_hid)
        self.t_max_1 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.t_drop_1 = nn.Dropout(p = drop_prob)
        self.t_ReLU_1 = nn.ReLU()

        #Conv2
        self.m_conv_2 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks2, padding = pad2)
        self.m_norm_2 = nn.BatchNorm1d(n_hid)
        self.m_max_2 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.m_drop_2 = nn.Dropout(p = drop_prob)
        self.m_ReLU_2 = nn.ReLU()
        self.m_norm_22 = nn.BatchNorm1d(n_hid)
        self.m_ReLU_22 = nn.ReLU()

        self.t_conv_2 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks2, padding = pad2)
        self.t_norm_2 = nn.BatchNorm1d(n_hid)
        self.t_max_2 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.t_drop_2 = nn.Dropout(p = drop_prob)
        self.t_ReLU_2 = nn.ReLU()


        #Conv3
        self.m_conv_3 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks3, padding = pad3)
        self.m_norm_3 = nn.BatchNorm1d(n_hid)
        self.m_max_3 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.m_drop_3 = nn.Dropout(p = drop_prob)
        self.m_ReLU_3 = nn.ReLU()

        self.t_conv_3 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks3, padding = pad3)
        self.t_norm_3 = nn.BatchNorm1d(n_hid)
        self.t_max_3 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.t_drop_3 = nn.Dropout(p = drop_prob)
        self.t_ReLU_3 = nn.ReLU()

        #Conv4
        self.m_conv_4 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks4, padding = pad4)
        self.m_norm_4 = nn.BatchNorm1d(n_hid)
        self.m_max_4 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.m_drop_4 = nn.Dropout(p = drop_prob)
        self.m_ReLU_4 = nn.ReLU()

        self.t_conv_4 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks4, padding = pad4)
        self.t_norm_4 = nn.BatchNorm1d(n_hid)
        self.t_max_4 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.t_drop_4 = nn.Dropout(p = drop_prob)
        self.t_ReLU_4 = nn.ReLU()

        #Conv5
        self.m_conv_5 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks5, padding = pad5)
        self.m_norm_5 = nn.BatchNorm1d(n_hid)
        self.m_max_5 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.m_drop_5 = nn.Dropout(p = drop_prob)
        self.m_ReLU_5 = nn.ReLU()

        self.t_conv_5 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks5, padding = pad5)
        self.t_norm_5 = nn.BatchNorm1d(n_hid)
        self.t_max_5 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.t_drop_5 = nn.Dropout(p = drop_prob)
        self.t_ReLU_5 = nn.ReLU()

        #Conv6
        self.m_conv_6 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks6, padding = pad6)
        self.m_norm_6 = nn.BatchNorm1d(n_hid)
        self.m_max_6 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.m_drop_6 = nn.Dropout(p = drop_prob)
        self.m_ReLU_6 = nn.ReLU()

        self.t_conv_6 = nn.Conv1d(in_channels = n_hid, out_channels = n_hid, kernel_size = ks6, padding = pad6)
        self.t_norm_6 = nn.BatchNorm1d(n_hid)
        self.t_max_6 = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.t_drop_6 = nn.Dropout(p = drop_prob)
        self.t_ReLU_6 = nn.ReLU()

        #Random projection module
        self.mrp_Linear = nn.Linear(in_features = (16)*n_hid, out_features = 16*n_hid)
        self.mrp_norm = nn.BatchNorm1d(16*n_hid)
        self.mrp_ReLU = nn.ReLU()

        self.trp_Linear = nn.Linear(in_features = (16)*n_hid, out_features = 16*n_hid)
        self.trp_norm = nn.BatchNorm1d(16*n_hid)
        self.trp_ReLU = nn.ReLU()

        self.mrp_prerpnorm = nn.BatchNorm1d(16*n_hid)
        self.trp_prerpnorm = nn.BatchNorm1d(16*n_hid)

        #Prediction module
        self.a_Linear0 = nn.Linear(in_features = 16*n_hid, out_features = 16*n_hid)
        self.a_BatchNorm = nn.BatchNorm1d(16*n_hid)
        self.a_ReLU = nn.ReLU(16*n_hid)

        self.a_Linear2 = nn.Linear(in_features = 16*n_hid, out_features = 2)

    def forward(self, x):
        bs0 = x.shape[0]

        x = x[:, :, features]

        if PS: print("Network sizes:\nInput:", x.shape)
        x = x.transpose(1, 2)
        if PS: print("Transposed X:", x.shape)

        pmhc = x[:, :, 0:192]
        tcr = x[:, :, 192:]

        if PS: print("Shapes MHC:p, TCR:", pmhc.shape, tcr.shape)

        pmhc0 = self.m_pad(pmhc)
        tcr0 = self.t_pad(tcr)
        if PS: print("Padded:", pmhc0.shape, tcr0.shape)

        pmhc = self.m_norm_1(self.m_conv_1(pmhc0))
        tcr = self.t_norm_1(self.t_conv_1(tcr0))

        pmhc0 = self.m_max_1(self.m_drop_1(self.m_ReLU_1(pmhc)))
        tcr0 = self.t_max_1(self.t_drop_1(self.t_ReLU_1(tcr)))
        if PS: print("Conv1", pmhc0.shape, tcr0.shape)

        pmhc1 = self.m_max_2(pmhc0 + self.m_drop_2(self.m_ReLU_2(self.m_norm_2(self.m_conv_2(pmhc0)))))
        tcr1 = self.t_max_2(tcr0 + self.t_drop_2(self.t_ReLU_2(self.t_norm_2(self.t_conv_2(tcr0)))))
        if PS: print("Conv2:", pmhc1.shape, tcr1.shape)

        pmhc0 = self.m_max_3(pmhc1 + self.m_drop_3(self.m_ReLU_3(self.m_norm_3(self.m_conv_3(pmhc1)))))
        tcr0 = self.t_max_3(tcr1 + self.t_drop_3(self.t_ReLU_3(self.t_norm_3(self.t_conv_3(tcr1)))))
        if PS: print("Conv3:", pmhc0.shape, tcr0.shape)

        pmhc1 = self.m_max_4(pmhc0 + self.m_drop_4(self.m_ReLU_4(self.m_norm_4(self.m_conv_4(pmhc0)))))
        tcr1 = self.t_max_4(tcr0 + self.t_drop_4(self.t_ReLU_4(self.t_norm_4(self.t_conv_4(tcr0)))))
        if PS: print("Conv4:", pmhc1.shape, tcr1.shape)

        pmhc0 = self.m_max_5(self.m_drop_5(self.m_ReLU_5(self.m_norm_5(self.m_conv_5(pmhc1)))))
        tcr0 = self.t_max_5(self.t_drop_5(self.t_ReLU_5(self.t_norm_5(self.t_conv_5(tcr1)))))
        if PS: print("Conv5:", pmhc0.shape, tcr0.shape)

        pmhc1 = self.m_max_6(self.m_drop_6(self.m_ReLU_6(self.m_norm_6(self.m_conv_6(pmhc0)))))
        tcr1 = self.t_max_6(self.t_drop_6(self.t_ReLU_6(self.t_norm_6(self.t_conv_6(tcr0)))))
        if PS: print("Conv6", pmhc1.shape, tcr1.shape)

        #Flattening (Pre-RP)
        pmhc1 = pmhc0
        tcr1 = tcr0

        pmhc = pmhc1.view(bs0, sz0)
        tcr = tcr1.view(bs0, sz0)
        if PS: print("Flattened view:", pmhc1.shape, tcr1.shape)

        pmhc = self.mrp_ReLU(self.mrp_norm(self.mrp_Linear(pmhc)))
        tcr = self.trp_ReLU(self.trp_norm(self.trp_Linear(tcr)))

        #Element Matrix product
        allparts = pmhc * tcr
        if PS: print("Matrix element multiplication:", allparts.shape)

        #Prediction module
        allparts = self.a_Linear2(allparts)
        if PS: print("Output:", allparts.shape)
        x = allparts
        ps = False

        return x
# Initialize network
#Setting number of features
features = list(range(0,142))

#Hyperparams for CNN
criterion = nn.CrossEntropyLoss()
in_channel = len(features)
n_hid = 56
batch_size = 32
drop_prob = 0.2

#Kernel sizes
ks0 = 1
pad0 = int((ks0) / 2)
ks1 = 11
pad1 = int((ks1) / 2)
ks2 = 9
pad2 = int((ks2) / 2)
ks3 = 7
pad3 = int((ks3) / 2)
ks4 = 7
pad4 = int((ks4) / 2)
ks5 = 5
pad5 = int((ks5) / 2)
ks6 = 3
pad6 = int((ks6) / 2)
PS = True
sz0 = 896
#bs0
net = Net().to(device)

# Loss and optimizer
criterion = nn.BCELoss() 
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

###############################
###         TRAIN           ###
###############################

print("Training")

num_epochs = 20 #100

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
cur_loss = 0
losses = []
loss_val = 0
val_losses = []

valid_acc_test = []

for epoch in range(num_epochs):
    print(epoch+1)
    cur_loss = 0
    val_loss = 0
    
    net.train()
    train_preds, train_targs = [], [] 
    for batch_idx, (data, target) in enumerate(train_ldr):
        X_batch =  data.float().clone().detach().requires_grad_(True)
        X_batch = X_batch.cuda()
        target_batch = torch.tensor(np.array(target), dtype = torch.float).unsqueeze(1)
        target_batch = target_batch.cuda()
        optimizer.zero_grad()
        output = net(X_batch)
        batch_loss = criterion(output, target_batch) 
        batch_loss.backward()
        optimizer.step()
        
        #preds = np.round(output.detach())
        preds = np.round(output.detach().cpu())
        train_targs += list(np.array(target_batch.cpu()))
        train_preds += list(preds.data.numpy())
        cur_loss += batch_loss   

    losses.append(cur_loss / len(X_train))
        
    
    net.eval()
    ### Evaluate validation
    val_preds, val_targs = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_ldr): ###
            x_batch_val = data.float().clone().detach()#.unsqueeze(1)
            x_batch_val = x_batch_val.cuda()
            y_batch_val = target.float().clone().detach().unsqueeze(1)
            y_batch_val = y_batch_val.cuda()
            output = net(x_batch_val).cpu()
            val_batch_loss = criterion(output, y_batch_val.cpu())
            #preds = np.round(output.detach())
            preds = np.round(output.detach().cpu())
            val_preds += list(preds.data.numpy()) 
            val_targs += list(np.array(y_batch_val.cpu()))
            val_loss += val_batch_loss  
            
        val_losses.append(val_loss / len(X_test))
        
        train_acc_cur = accuracy_score(train_targs, train_preds)  
        valid_acc_cur = accuracy_score(val_targs, val_preds) 

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)

epoch = np.arange(1,len(train_acc)+1)
#plt.figure()
#plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
#plt.legend(['Train Accucary','Validation Accuracy'])
#plt.xlabel('Epoch'), plt.ylabel('Acc')
#plt.savefig('/home/projects/ht3_aim/people/alsj/cd4cd8/data/figures/alpha/CNN_Accuracy.png')


epoch = np.arange(1,len(train_acc)+1)
#plt.figure()
#plt.plot(epoch, losses, 'r', epoch, val_losses, 'b')
#plt.legend(['Train Loss','Validation Loss'])
#plt.xlabel('Epoch'), plt.ylabel('Loss')
#plt.savefig('/home/projects/ht3_aim/people/alsj/cd4cd8/data/figures/alpha/CNN_Loss.png')

print("Train accuracy:")
print(train_acc)
print("Validation accuracy:")
print(valid_acc)

print("Confusion matrix train:")
print(confusion_matrix(train_targs, train_preds))
print("Confusion matrix test:")
print(confusion_matrix(val_targs, val_preds))

# ROC
fpr, tpr, threshold = metrics.roc_curve(train_targs, train_preds)
roc_auc = metrics.auc(fpr, tpr)

# plot ROC
print("Train ROC:")
import matplotlib.pyplot as plt
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()

# ROC
fpr, tpr, threshold = metrics.roc_curve(val_targs, val_preds)
roc_auc = metrics.auc(fpr, tpr)

# plot ROC
print("Validation ROC")
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()



