#!/usr/bin/env python
# coding: utf-8

# Imports

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters

###############################
###    Load data            ###
###############################

X_train = np.load("hackathon_data/P1_input.npz")["arr_0"]
y_train = np.load("hackathon_data/P1_labels.npz")["arr_0"]
nsamples, nx, ny = X_train.shape
print(nsamples,nx,ny)

X_test = np.load("hackathon_data/P2_input.npz")["arr_0"]
y_test = np.load("hackathon_data/P2_labels.npz")["arr_0"]
nsamples, nx, ny = X_test.shape
print(nsamples,nx,ny)

print("Percent positive samples in train:")
print(len(y_train[y_train == 1])/len(y_train)*100)
print("Percent positive samples in test:")
print(len(y_test[y_test == 1])/len(y_test)*100)

# make the data set into one dataset that can go into dataloader
train_ds = []
for i in range(len(X_train)):
    train_ds.append([np.transpose(X_train[i]), y_train[i]])

test_ds = []
for i in range(len(X_test)):
    test_ds.append([np.transpose(X_test[i]), y_test[i]])

bat_size = 128
train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)

# Set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

###############################
###    Define network       ###
###############################

print("Initializing network")

# Hyperparameters
input_size = 420
num_classes = 1
learning_rate = 0.01
class Net(nn.Module):
    def __init__(self,  num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=54, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(100)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(50)
        self.fc1 = nn.Linear(1300,225)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(225,10)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(10, num_classes)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
# Initialize network
net = Net(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.BCELoss() 
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

###############################
###         TRAIN           ###
###############################

print("Training")

num_epochs = 5

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
losses = []
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
        X_batch = X_batch
        target_batch = torch.tensor(np.array(target), dtype = torch.float).unsqueeze(1)
        target_batch = target_batch
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
            x_batch_val = x_batch_val
            y_batch_val = target.float().clone().detach().unsqueeze(1)
            y_batch_val = y_batch_val
            output = net(x_batch_val)
            val_batch_loss = criterion(output, y_batch_val)
            #preds = np.round(output.detach())
            preds = np.round(output.detach())
            val_preds += list(preds.data.numpy()) 
            val_targs += list(np.array(y_batch_val))
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

epoch = np.arange(1,len(train_acc)+1)
#plt.figure()
#plt.plot(epoch, losses, 'r', epoch, val_losses, 'b')
#plt.legend(['Train Loss','Validation Loss'])
#plt.xlabel('Epoch'), plt.ylabel('Loss')

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
#print("Train ROC:")
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
#print("Validation ROC")
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()



