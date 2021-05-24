#!/usr/bin/env python
# coding: utf-8

# Imports
import numpy as np
import os
import sys
import glob
#import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef, recall_score, precision_score
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
import torchvision
from sklearn.preprocessing import normalize

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################
###    Load data            ###
###############################

data_list = []
target_list = []
origin_list = []

for fp in glob.glob("/home/projects/ht3_aim/people/idamei/data/train_data2/*input.npz"):
    data = np.load(fp)["arr_0"]
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    origins = np.load(fp.replace("input", "origin"))["arr_0"]

    data_list.append(data)
    target_list.append(targets)
    origin_list.append(origins)

# Which features and residues to use

# All features
#features = list(range(54))
# Only one-hot
#features = list(range(0,20))
# Only energy terms
#features = list(range(20,54))
# One-hot and global energy terms
#features = list(range(20)) + list(range(27,54))
# One-hot and per-residue energy terms
#features = list(range(27))
# Only per-residue energy terms
#features = list(range(20,27))
# One-hot and total scores
#features = list(range(20)) + [27, 41, 48]
# Only global energy terms
#features = list(range(27,54))

#features = list(range(1,54))
features = [4]

# Whole sequence
residues = list(range(416))
# Without MHC
#residues = list(range(179,416))
# Without peptide
#residues = list(range(179)) + list(range(188,416))
# Without TCR
#residues = list(range(188))

###############################
###     Define network      ###
###############################

print("Initializing network")

# Hyperparameters
input_size = len(residues)
print("Input size: ", input_size)
num_classes = 1
learning_rate = 0.001
bat_size = 128
n_features = len(features)
criterion = nn.BCEWithLogitsLoss()
class Net(nn.Module):
    def __init__(self,  num_classes):
        super(Net, self).__init__()
        self.bn0 = nn.BatchNorm1d(n_features)    
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(100)
        
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(100)
        
        self.rnn = nn.LSTM(input_size=100,hidden_size=26,num_layers=3, dropout=0.1, batch_first=True, bidirectional = True)
        self.drop = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(26*2, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.bn0(x)  
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.drop(x)
        x = x.transpose_(2, 1)
        x, (h, c) = self.rnn(x)
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        cat = self.drop(cat)
        x = self.fc1(cat)
        return x

# Weighted loss function from one of the hackathon teams
def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))

###############################
###         TRAIN           ###
###############################

def train(train_ldr, val_ldr, test_ldr):

    num_epochs = 100

    train_acc = []
    valid_acc = []
    train_losses = []
    valid_losses = []
    train_auc = []
    valid_auc = []

    no_epoch_improve = 0
    min_val_loss = np.Inf
    
    test_probs, test_preds, test_targs = [], [], []

    for epoch in range(num_epochs):
        cur_loss = 0
        val_loss = 0
        # Train
        net.train()
        train_preds, train_targs, train_probs = [], [], []
        for batch_idx, (data, target) in enumerate(train_ldr):
            X_batch =  data.float().detach().requires_grad_(True).cuda()
            target_batch = torch.tensor(np.array(target), dtype = torch.float).unsqueeze(1).cuda()

            optimizer.zero_grad()
            output = net(X_batch)

            #batch_loss = weighted_binary_cross_entropy(output, target_batch, [0.75,0.25])
            batch_loss = criterion(output, target_batch)
            batch_loss.backward()
            optimizer.step()
      
            probs = torch.sigmoid(output.detach())
            preds = np.round(probs.cpu())
            train_probs += list(probs.data.cpu().numpy())
            train_targs += list(np.array(target_batch.cpu()))
            train_preds += list(preds.data.numpy())
            cur_loss += batch_loss.detach()

        train_losses.append(cur_loss / len(train_ldr.dataset))        

        net.eval()
        # Validation
        val_preds, val_targs, val_probs = [], [], []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_ldr):
                x_batch_val = data.float().detach().cuda()
                y_batch_val = target.float().detach().unsqueeze(1).cuda()

                output = net(x_batch_val)
                val_batch_loss = criterion(output, y_batch_val)

                probs = torch.sigmoid(output.detach())
                preds = np.round(probs.cpu())
                val_probs += list(probs.data.cpu().numpy())
                val_preds += list(preds.data.numpy()) 
                val_targs += list(np.array(y_batch_val.cpu()))
                val_loss += val_batch_loss.detach()

            valid_losses.append(val_loss / len(val_ldr.dataset))
            print("\nEpoch:", epoch+1)

            train_acc_cur = accuracy_score(train_targs, train_preds)  
            valid_acc_cur = accuracy_score(val_targs, val_preds) 
            train_auc_cur = roc_auc_score(train_targs, train_probs)
            valid_auc_cur = roc_auc_score(val_targs, val_probs)

            train_acc.append(train_acc_cur)
            valid_acc.append(valid_acc_cur)
            train_auc.append(train_auc_cur)
            valid_auc.append(valid_auc_cur)

            print("Training loss:", train_losses[-1].item(), "Validation loss:", valid_losses[-1].item())
            print("MCC Train:", matthews_corrcoef(train_targs, train_preds), "MCC val:", matthews_corrcoef(val_targs, val_preds))
            print("AUC Train:", train_auc_cur, "AUC val:", valid_auc_cur)
            print("Confusion train:")
            print(confusion_matrix(train_targs, train_preds))
            print("Confusion validation:")
            print(confusion_matrix(val_targs, val_preds))

        # Early stopping
        if (val_loss / len(X_valid)).item() < min_val_loss:
            no_epoch_improve = 0
            min_val_loss = (val_loss / len(X_valid))
        else:
            no_epoch_improve +=1
        if no_epoch_improve == 7:
            print("Early stopping\n")
            break
            
    # Test
    if test_ldr is not None:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_ldr): ###
                x_batch_test = data.float().detach().cuda()
                y_batch_test = target.float().detach().unsqueeze(1).cuda()

                output = net(x_batch_test)

                #test_batch_loss = criterion(output, y_batch_test)
                probs = torch.sigmoid(output.detach())
                preds = np.round(probs.cpu())
                test_probs += list(probs.data.cpu().numpy())
                test_preds += list(preds.data.numpy())
                test_targs += list(np.array(y_batch_test.cpu()))

    return train_acc, train_losses, train_auc, valid_acc, valid_losses, valid_auc, test_probs, test_preds, test_targs

###############################
### Nested cross-validation ###
###############################

test_probs_all = [None] * 20
test_preds_all = [None] * 20
test_targs_all = [None] * 20

run_count = 0
for i in range(5):
    
    # Test set
    X_test = data_list[i]
    print("Length before removing swapped:", len(X_test))
    X_test = X_test[origin_list[i] != 1] # remove swapped   (0:positive, 1:swapped, 2:10x)
    print("Length after removing swapped:", len(X_test))
    y_test = target_list[i]
    y_test = y_test[origin_list[i] != 1] # remove swapped
    test_ds = []
    for k in range(len(X_test)):
        X_test_k = X_test[k][residues, :]
        X_test_k = X_test_k[:, features]
        test_ds.append([np.transpose(X_test_k), y_test[k]])
    test_ldr = torch.utils.data.DataLoader(test_ds,batch_size=bat_size, shuffle=True)

    X_remaining = data_list[:i] + data_list[i+1:]
    y_remaining = target_list[:i] + target_list[i+1:]

    for j in range(4):
        
        #print(f"Run {run_count+1}. Testing on partition {i+1}.")
        print("Run {}. Testing on partition {}.".format(run_count+1, i+1))
        
        # Validation set
        X_valid = X_remaining[j]
        y_valid = y_remaining[j]
        valid_ds = []
        for k in range(len(X_valid)):
            X_valid_k = X_valid[k][residues, :]
            X_valid_k = X_valid_k[:, features]
            valid_ds.append([np.transpose(X_valid_k), y_valid[k]])
        valid_ldr = torch.utils.data.DataLoader(valid_ds,batch_size=bat_size, shuffle=True)
        
        # Train set
        X_train = np.concatenate(X_remaining[:j] + X_remaining[j+1:])
        y_train = np.concatenate(y_remaining[:j] + y_remaining[j+1:])
        train_ds = []
        for k in range(len(X_train)):
            X_train_k = X_train[k][residues, :]
            X_train_k = X_train_k[:, features]
            train_ds.append([np.transpose(X_train_k), y_train[k]])
        train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)
            
        # Initialize network
        net = Net(num_classes=num_classes).to(device)

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                       weight_decay=0.0005,
                       amsgrad=True)

        # Train
        _, _, _, _, _, _, test_probs, test_preds, test_targs = train(train_ldr, valid_ldr, test_ldr)
        test_probs_all[run_count] = test_probs
        test_preds_all[run_count] = test_preds
        test_targs_all[run_count] = test_targs
        
        run_count += 1


##############################################
### Performance on nested cross-validation ###
##############################################

print("\nTotal performance:\n")

test_targs_all_concatenated = [item for sublist in test_targs_all for item in sublist]
test_probs_all_concatenated = [item for sublist in test_probs_all for item in sublist]
test_preds_all_concatenated = [item for sublist in test_preds_all for item in sublist]

print("AUC:")
print(roc_auc_score(test_targs_all_concatenated, test_probs_all_concatenated))

print("MCC:")
print(matthews_corrcoef(test_targs_all_concatenated, test_preds_all_concatenated))

print("Confusion matrix:")
print(confusion_matrix(test_targs_all_concatenated, test_preds_all_concatenated))

print("Accuracy:")
print(accuracy_score(test_targs_all_concatenated, test_preds_all_concatenated))

print("Precission:")
print(precision_score(test_targs_all_concatenated, test_preds_all_concatenated))

print("Recall:")
print(recall_score(test_targs_all_concatenated, test_preds_all_concatenated))

print("f1_score:")
print(f1_score(test_targs_all_concatenated, test_preds_all_concatenated))

print("Classificaion report:")
print(classification_report(test_targs_all_concatenated, test_preds_all_concatenated))


#print("\nPerformance for each run:\n")

#for i in range(20):

#    print(f"Performance run {i+1}")
#
#    print("AUC:")
#    print(roc_auc_score(test_targs_all[i], test_probs_all[i]))
#
#    print("MCC:")
#    print(matthews_corrcoef(test_targs_all[i], test_preds_all[i]))
#
#    print("Confusion matrix:")
#    print(confusion_matrix(test_targs_all[i], test_preds_all[i]))
#
#    print("Accuracy:")
#    print(accuracy_score(test_targs_all[i], test_preds_all[i]))
#
#    print("Classificaion report:")
#    print(classification_report(test_targs_all[i], test_preds_all[i]))

def plot_roc(targets, predictions):
    # ROC
    fpr, tpr, threshold = metrics.roc_curve(targets, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    # plot ROC
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()


plot_roc(test_targs_all_concatenated, test_probs_all_concatenated)
plt.title("Test AUC all runs concatenated")

sys.exit()

#################################
###       Simple train        ###
#################################

# Train on 4, early stopping on the 5th

X_train = np.concatenate(data_list[ :-1])
y_train = np.concatenate(target_list[:-1])
nsamples, nx, ny = X_train.shape
print("Training set shape:", nsamples,nx,ny)

X_valid = np.concatenate(data_list[-1: ])
y_valid = np.concatenate(target_list[-1: ])
nsamples, nx, ny = X_valid.shape
print("Validation set shape:", nsamples,nx,ny)

# Dataloader
train_ds = []
for i in range(len(X_train)):
    #X_array = np.transpose(X_train[i][:,:])
    #normalized_X_array = normalize(X_array, axis=1, norm='l1')
    #y_value = y_train[i]
    #train_ds.append([normalized_X_array, y_value])
    train_ds.append([np.transpose(X_train[i][:,features]), y_train[i]])

val_ds = []
for i in range(len(X_valid)):
    #X_array = np.transpose(X_val[i][:,:])
    #normalized_X_array = normalize(X_array, axis=1, norm='l1')
    #y_value = y_val[i]
    #val_ds.append([normalized_X_array, y_value])
    val_ds.append([np.transpose(X_valid[i][:,features]), y_valid[i]])

train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)
val_ldr = torch.utils.data.DataLoader(val_ds,batch_size=bat_size, shuffle=True)

# Initialize network
net = Net(num_classes=num_classes).to(device)

### Optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                       weight_decay=0.0005,
                       amsgrad=True,
                       )
#optim.SGD(net.parameters(), lr=learning_rate)

train_acc, train_losses, train_auc, valid_acc, valid_losses, valid_auc, _, _ = train(train_ldr, val_ldr, None)


#####################################
###  PERFORMANCE ON SIMPLE TRAIN  ###
#####################################

epoch = np.arange(1,len(train_losses)+1)
plt.figure()
plt.plot(epoch, train_losses, 'r', epoch, valid_losses, 'b')
plt.legend(['Train Loss','Validation Loss'])
plt.xlabel('Epoch'), plt.ylabel('Loss')

epoch = np.arange(1,len(train_auc)+1)
plt.figure()
plt.plot(epoch, train_auc, 'r', epoch, valid_auc, 'b')
plt.legend(['Train AUC','Validation AUC'])
plt.xlabel('Epoch'), plt.ylabel('AUC')

epoch = np.arange(1,len(train_acc)+1)
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Accuracy','Validation Accuracy'])
plt.xlabel('Epoch'), plt.ylabel('Acc')

print('Training Classification Report')
print(classification_report(train_targs, train_preds))

print('Validation Classification Report')
print(classification_report(val_targs, val_preds))

plot_roc(train_targs, train_probs)
plt.title("Training AUC")
plot_roc(val_targs, val_probs)
plt.title("Validation AUC")

