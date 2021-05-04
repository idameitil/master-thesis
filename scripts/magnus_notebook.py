#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
# Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters


# ## Model

# In[2]:


###############################
###    Load data            ###
###############################

data_list = []
target_list = []

import glob
for fp in glob.glob("/home/ida/tcr-pmhc/data/train/*input.npz"):
    data = np.load(fp)["arr_0"]
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    
    data_list.append(data)
    target_list.append(targets)


# In[3]:



# Note:
# Choose your own training and val set based on data_list and target_list
# Here using the last partition as val set

#X_train = np.concatenate(data_list[0:])
#y_train = np.concatenate(target_list[0:]) #taking full data to train
X_train = np.concatenate(data_list[ :-1])
y_train = np.concatenate(target_list[:-1])
#X_train = np.concatenate(data_list[0,2,3])
#y_train = np.concatenate(target_list[0,2,3])
nsamples, nx, ny = X_train.shape
print("Training set shape:", nsamples,nx,ny)

X_val = np.concatenate(data_list[-1: ])
y_val = np.concatenate(target_list[-1: ])
#X_val = data_list[1]
#y_val = target_list[1]

#print(X_val.shape)
nsamples, nx, ny = X_val.shape
print("val set shape:", nsamples,nx,ny)

p_neg = len(y_train[y_train == 1])/len(y_train)*100
print("Percent positive samples in train:", p_neg)

p_pos = len(y_val[y_val == 1])/len(y_val)*100
print("Percent positive samples in val:", p_pos)

# make the data set into one dataset that can go into dataloader
train_ds = []
for i in range(len(X_train)):
    train_ds.append([np.transpose(X_train[i]), y_train[i]])

val_ds = []
for i in range(len(X_val)):
    val_ds.append([np.transpose(X_val[i]), y_val[i]])

bat_size = 128
print("\nNOTE:\nSetting batch-size to", bat_size)
train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)
val_ldr = torch.utils.data.DataLoader(val_ds,batch_size=bat_size, shuffle=True)


# Set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device (CPU/GPU):", device)
device = torch.device("cpu")


# In[4]:


print("Using device (CPU/GPU):", device)


# In[5]:


################################ TRYING CNN LSTM HYBRID MODELS
###############################
###    Define network       ###
###############################

print("Initializing network")

# Hyperparameters
input_size = 420
num_classes = 1
learning_rate = 0.001 #0.0001
PS = False
# try normalizing before
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

        self.a_Linear2 = nn.Linear(in_features = 16*n_hid, out_features = 1)

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

        return x
    
features = list(range(0,142))
criterion = nn.BCEWithLogitsLoss() 
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

net = Net().to(device)


# In[6]:


# summary(net, (54, 420))


# In[7]:


#####weighted loss
def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) +                weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))


# In[8]:


###############################
###         TRAIN           ###
###############################

print("Training")

num_epochs = 30

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
losses = []
val_losses = []

for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    
    net.train()
    train_preds, train_targs = [], [] 
    for batch_idx, (data, target) in enumerate(train_ldr):
        X_batch =  data.float().detach().requires_grad_(True)
        target_batch = torch.tensor(np.array(target), dtype = torch.float).unsqueeze(1)
        
        optimizer.zero_grad()
        output = net(X_batch)
        

        #batch_loss = weighted_binary_cross_entropy(output, target_batch, [0.75,0.25])
        batch_loss = criterion(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        preds = np.round(output.detach().cpu())
        train_targs += list(np.array(target_batch.cpu()))
        train_preds += list(preds.data.numpy().flatten())
        cur_loss += batch_loss.detach()

    losses.append(cur_loss / len(train_ldr.dataset))        
    
    net.eval()
    ### Evaluate validation
    val_preds, val_targs = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_ldr): ###
            x_batch_val = data.float().detach()
            y_batch_val = target.float().detach().unsqueeze(1)
            
            output = net(x_batch_val)
            

            val_batch_loss = weighted_binary_cross_entropy(output, y_batch_val, [0.75,0.25])
            #val_batch_loss = criterion(output, y_batch_val)
            
            preds = np.round(output.detach())
            val_preds += list(preds.data.numpy().flatten()) 
            val_targs += list(np.array(y_batch_val))
            val_loss += val_batch_loss.detach()
            
        val_losses.append(val_loss / len(val_ldr.dataset))
        print("\nEpoch:", epoch+1)
        
        train_acc_cur = accuracy_score(train_targs, train_preds)  
        valid_acc_cur = accuracy_score(val_targs, val_preds) 

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        
        from sklearn.metrics import matthews_corrcoef
        print("Training loss:", losses[-1].item(), "Validation loss:", val_losses[-1].item(), end = "\n")
        print("MCC Train:", matthews_corrcoef(train_targs, train_preds), "MCC val:", matthews_corrcoef(val_targs, val_preds))
        print("Confusion train:")
        print(confusion_matrix(train_targs, train_preds))
        print("Confusion validation:")
        print(confusion_matrix(val_targs, val_preds))
        ##save model at every epoch
        #torch.save(net.state_dict(), os.path.join(model_dir, 'model-epoch-{}.pt'.format(epoch)))
        #print('Models saved at epoch', epoch+1)


# ## MH

# In[77]:


###############################
###        PERFORMANCE      ###
###############################

epoch = np.arange(1,len(train_acc)+1)
plt.figure()
plt.plot(epoch, losses, 'r', epoch, val_losses, 'b')
plt.legend(['Train Loss','Validation Loss'])
plt.xlabel('Epoch'), plt.ylabel('Loss')

epoch = np.arange(1,len(train_acc)+1)
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Accuracy','Validation Accuracy'])
plt.xlabel('Epoch'), plt.ylabel('Acc')

print("Train accuracy:", train_acc, sep = "\n")
print("Validation accuracy:", valid_acc, sep = "\n")


# In[78]:


#### performance
from sklearn.metrics import matthews_corrcoef
print("MCC Train:", matthews_corrcoef(train_targs, train_preds))
print("MCC Test:", matthews_corrcoef(val_targs, val_preds))

print("Confusion matrix train:", confusion_matrix(train_targs, train_preds), sep = "\n")
print("Confusion matrix test:", confusion_matrix(val_targs, val_preds), sep = "\n")


# In[79]:


from sklearn.metrics import classification_report






# In[80]:


print('Training Classification Report')
print(classification_report(train_targs, train_preds))


# In[ ]:





# In[81]:


print('Validation Classification Report')
print(classification_report(val_targs, val_preds))


# In[82]:


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


# In[83]:


plot_roc(train_targs, train_preds)
plt.title("Training AUC")
plot_roc(val_targs, val_preds)
plt.title("Validation AUC")


# In[114]:


from torchviz import make_dot


# In[133]:


# Initialize network
net = Net(num_classes=num_classes).to(device)
x = torch.randn(1,54,420)
#y = model_see(x)

vis_graph = make_dot(net(x).mean(), params=dict(net.named_parameters()))
# vis_graph.view()


# In[136]:


from graphviz import Source; 
model_arch = make_dot(net(x).mean(), params=dict(net.named_parameters())); 
Source(model_arch).render('arch.png')


# In[137]:


from graphviz import Source
Source.from_file('/content/Digraph.gv')


# In[ ]:





# In[ ]:





# In[ ]:


make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz",


# In[132]:


print(vis_graph)
#vis_graph.format = 'png'
#vis_graph.render("arch.png")


# In[ ]:





# In[ ]:





# In[ ]:


#model save below
# Write model to disk for use in predict.py
print("Saving model to src/model.pt")
torch.save(net.state_dict(), "/content/tcr-pmhc/src/model.pt")


# In[ ]:


#### download the zip
get_ipython().system('zip -r /content/models.zip /content/models_save')


# In[ ]:





# In[ ]:





# # Helpful scripts

# # Show dataset as copied dataframes with named features
# The dataset is a 3D numpy array, of dimensions n_complexes x features x positions. This makes viewing the features for individual complexes or samples challenging. Below is a function which copies the entire dataset, and converts it into a list of DataFrames with named indices and columns, in order to make understanding the data easier.
# 
# NB: This list of dataframes are only copies, and will not be passable into the neural network architecture.

# In[ ]:


pd.read_csv("/content/tcr-pmhc/data/example.csv")


# In[ ]:


def copy_as_dataframes(dataset_X):
    """
    Returns list of DataFrames with named features from dataset_X,
    using example CSV file
    """
    df_raw = pd.read_csv("/content/tcr-pmhc/data/example.csv")
    return [pd.DataFrame(arr, columns = df_raw.columns) for arr in dataset_X]

named_dataframes = copy_as_dataframes(X_train)
print("Showing first complex as dataframe. Columns are positions and indices are calculated features")
named_dataframes[0]


# # View complex MHC, peptide and TCR alpha/beta sequences
# You may want to view the one-hot encoded sequences as sequences in single-letter amino-acid format. The below function will return the TCR, peptide and MHC sequences for the dataset as 3 lists.

# In[ ]:


def oneHot(residue):
    """
    Converts string sequence to one-hot encoding
    Example usage:
    seq = "GSHSMRY"
    oneHot(seq)
    """
    
    mapping = dict(zip("ACDEFGHIKLMNPQRSTVWY", range(20)))
    if residue in "ACDEFGHIKLMNPQRSTVWY":
        return np.eye(20)[mapping[residue]]
    else:
        return np.zeros(20)
def reverseOneHot(encoding):
    """
    Converts one-hot encoded array back to string sequence
    """
    mapping = dict(zip(range(20),"ACDEFGHIKLMNPQRSTVWY"))
    seq=''
    for i in range(len(encoding)):
        if np.max(encoding[i])>0:
            seq+=mapping[np.argmax(encoding[i])]
    return seq

def extract_sequences(dataset_X):
    """
    Return DataFrame with MHC, peptide and TCR a/b sequences from
    one-hot encoded complex sequences in dataset X
    """
    mhc_sequences = [reverseOneHot(arr[0:179,0:20]) for arr in dataset_X]
    pep_sequences = [reverseOneHot(arr[179:190,0:20]) for arr in dataset_X]
    tcr_sequences = [reverseOneHot(arr[192:,0:20]) for arr in dataset_X]
    df_sequences = pd.DataFrame({"MHC":mhc_sequences, "peptide":pep_sequences,
                                 "tcr":tcr_sequences})
    return df_sequences


# In[ ]:


complex_sequences = extract_sequences(X_val)
print("Showing MHC, peptide and TCR alpha/beta sequences for each complex")
complex_sequences


# In[ ]:





# In[ ]:


### backup of convnet
# ################################ THIS NETWORK WORKS GOOD- DO NOT EDIT
# ###############################
# ###    Define network       ###
# ###############################

# print("Initializing network")

# # Hyperparameters
# input_size = 420
# num_classes = 1
# learning_rate = 0.001 #0.0001

# class Net(nn.Module):
#     def __init__(self,  num_classes):
#         super(Net, self).__init__()
#         self.bn0 = nn.BatchNorm1d(54) ### edited after 4:30 PM       
#         self.conv1 = nn.Conv1d(in_channels=54, out_channels=100, kernel_size=3, stride=2, padding=1)
#         torch.nn.init.kaiming_uniform_(self.conv1.weight)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv1_bn = nn.BatchNorm1d(100)
        
#         self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1)
#         torch.nn.init.kaiming_uniform_(self.conv2.weight)
#         self.conv2_bn = nn.BatchNorm1d(100)
        
#         self.fc1 = nn.Linear(2600, num_classes)
#         torch.nn.init.xavier_uniform_(self.fc1.weight)
        
#     def forward(self, x):
#         x = self.bn0(x)      ### edited after 4:30 PM   
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.conv1_bn(x)
        
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.conv2_bn(x)
        
#         x = x.view(x.size(0), -1)
#         #x = self.fc1(x)
#         x = torch.sigmoid(self.fc1(x))

        
#         return x
    
# # Initialize network
# net = Net(num_classes=num_classes).to(device)


# ## unbalanced loss
# #unbal_weight = torch.tensor([1,0.333]).to(device)

# # Loss and optimizer
# #criterion = nn.BCELoss()
# #criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(net.parameters(), lr=learning_rate,
#                        weight_decay=0.003,
#                        amsgrad=True,
#                        ) 



# #optim.SGD(net.parameters(), lr=learning_rate)


# In[ ]:




