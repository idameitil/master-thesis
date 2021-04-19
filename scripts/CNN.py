print("Importing modules...")
import optparse, os, sys, glob, re, pickle, time, datetime
import numpy as np
import pandas as pd

from fastai.basic_data import DatasetType, DataBunch
from fastai.basic_train import Learner, partial
from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from fastai.train import fit_one_cycle

import torch
import torch.nn as nn
import torch.utils.data as tdatautils

from sklearn.metrics import f1_score, confusion_matrix, auc, roc_auc_score, matthews_corrcoef, average_precision_score

from allscripts import upsample, data_generator, generate_weights, pad, load_data

print("Importing done")

#############################
# Functions
#############################

#Convert numpy to torch
def to_torch_data(x,np_type,tch_type):
    return torch.from_numpy(x.astype(np_type)).to(tch_type)

#Main function for calculating model performance
def calculate_performance_measures(y_scores, yhat, targets):

    y_true = np.array(targets)
    y_scores_binary = np.where(y_scores > 0.5, 1, 0)

    correct = yhat == y_true
    accuracy = round(sum(correct) / len(targets), 3)
    auc = round(roc_auc_score(y_true, y_scores), 3)
    mcc = round(matthews_corrcoef(y_true, y_scores_binary), 3)
    avp = round(average_precision_score(y_true, y_scores), 3)

    confusion = confusion_matrix(y_true, yhat)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    tpr = round(tp / (tp+fn), 3)
    tnr = round(tn / (tn+fp), 3)
    precision = round(tp / (tp + fp), 3)
    recall = round(tp / (tp + fn), 3)

    return (accuracy, auc, mcc, avp, tpr, tnr, precision, recall, confusion)

# Save performance sets, hyperparam and comment to CSVfile
now = time.time()
def stats_to_csv(start_time = time.time(), val_part = 4, test_part = int(), comment = "", LR="", ds=DatasetType.Valid):
    train_str = [0, 1, 2, 3, 4]
    val_str = val_part
    test_str = test_part

    stat_df = pd.DataFrame(columns = ["Comment", "Test", "Validation", "Training", "Accuracy", "AUC", "MCC", "AVP", "TPR", "TNR", "Prec", "Rec", "Confusion matrix", "LR", "Duration (s)", "Timestamp"])

    #Check for CSV file
    if not glob.glob(CSVFILE):
        stat_df.to_csv(CSVFILE, mode = "w", header = True, index = False)

    #Remove val / test from training parts
    train_str.remove(val_str)
    if test_part != int():
        train_str.remove(test_str)

    #Get raw predictions
    preds = learn.get_preds(ds)
    outputs = preds[0]
    targets = preds[1]

    #Find highest multi-class prediction (yes, this is wrong ...)
    yhat = []
    for i in range(len(outputs)):
        pred = outputs[i].tolist()
        pred = pred.index(max(pred))
        yhat.append(pred)
    yhat = np.array(yhat)
    y_scores = outputs[:, 1]

    # Calculate performance measures
    accuracy, auc, mcc, avp, tpr, tnr, precision, recall, confusion = calculate_performance_measures(y_scores, yhat, targets)

    duration = round(time.time() - start_time)
    timestamp = str(datetime.datetime.now())

    #Add to stat_df and save to CSV
    row = [comment, test_str, val_str, train_str, accuracy, auc, mcc, avp, tpr, tnr, precision, recall, confusion, LR, duration, timestamp]
    stat_df.loc[len(stat_df)] = row
    stat_df.to_csv(CSVFILE, mode = "a", header = False, index = True)
    print(["ACC", "AUC", "MCC", "AVP", "TPR", "TNR", "Prec", "Rec", "Confusion matrix"])
    print(accuracy, auc, mcc, avp, tpr, tnr, precision, recall)
    print(confusion)

def stats_to_csv_nested(outputs, targets, start_time = time.time(), val_part = 4, test_part = int(), comment = "", ds=DatasetType.Valid):
    train_str = [0, 1, 2, 3, 4]
    val_str = val_part
    test_str = test_part

    stat_df = pd.DataFrame(columns = ["Comment", "Test", "Validation", "Training", "Accuracy", "AUC", "MCC", "AVP", "TPR", "TNR", "Precision", "recall", "Confusion matrix", "Duration (s)", "Timestamp"])

    #Check for CSV file
    if not glob.glob("check.csv"):
        stat_df.to_csv("check.csv", mode = "w", header = True, index = False)

    #Remove val / test from training parts
    train_str.remove(val_str)
    if test_part != int():
        train_str.remove(test_str)

    #Get model performance
    #Find highest multi-class prediction (yes, this is wrong ...)
    yhat = np.array(outputs[0:len(targets)])
    y_scores = outputs
    accuracy, auc, mcc, avp, tpr, tnr, precision, recall, confusion = calculate_performance_measures(y_scores, yhat, targets)

    duration = round(time.time() - start_time)
    timestamp = str(datetime.datetime.now())

    #Add to stat_df and save to CSV
    row = [comment, test_str, val_str, train_str, accuracy, auc, mcc, avp, tpr, tnr, precision, recall, confusion, duration, timestamp]
    stat_df.loc[len(stat_df)] = row
    stat_df.to_csv("check.csv", mode = "a", header = False, index = True)
    print(row)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

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

def simple_train(cycles=4, LR=10*[1e-02, 1e-01], epochs=10*[3],
                         val_part = int(4), test_part = str(),
                         skip = False):
    """Model train function. Cross validation with specified LR, epochs per cycle and number of cycles."""

    train_part=[0, 1, 2, 3, 4]
    LR = pd.Series(LR)

    #Skip validation
    if skip == True:
        learn.data.valid_dl = None

    #Train model
    for i in range(0, cycles):
        print("Cycle:", i+1, "/", cycles, "Epochs:", epochs[i], "LR:", LR[i*2], "->", LR[(i*2)+1])
        now = time.time()
        learn.fit_one_cycle(epochs[i], max_lr=slice(None, LR[i*2], LR[(i*2)+1]), wd = 0.01)

    #Save model to filepath
    train_part.remove(val_part)
    if test_part == str():
        test_part = "X"
    else:
        train_part.remove(test_part)

    test_str = str(test_part)
    val_str = str(val_part)
    train_str = "".join(map(str, train_part))

    filepath = OUTDIR + "T" + test_str + "V" + val_str + "_" + train_str
    learn.save(filepath)

    return(filepath)

#############################
# MAIN
#############################

#Set random seeds
torch.cuda.manual_seed_all(1)
np.random.seed(1)

# Set working directory to script path
abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

#############################
# Parse commandline options
#############################

parser = optparse.OptionParser()

#Set mode, and random or ordered partitions
parser.add_option("-m", "--mode", dest="MODE", default=1, help="Set training mode: 1. Cross-val, 2. Nested-cross-val, .")
parser.add_option("--r", "--random", dest="RANDOM", default=False, help="Set (random) partitions from filenames with True. Default ordered partitions")
parser.add_option("--rp", dest="RP", default="False", help="Whether to use random projection unit")
parser.add_option("--sets", dest="SETS", default=4, help="Number of times to train the network (e.g. 2 sets of 4 cycles)")
parser.add_option("--gpu", dest="GPU", default="True", help="Whether script is being run on a GPU")

#Set outdir, comment and csvfile path
parser.add_option("-o", "--outdir", dest="OUTDIR", default="../results/0415_m1/", help="Set number of features used for each sequence position in X input data")
parser.add_option("-c", dest="COMMENT", default="", help="Comment for CSV file")
parser.add_option("--csvfile", dest="CSVFILE", default="../results/0415_m1/0415_m1.csv")

#Network parameters
parser.add_option("--dp", dest="DP", default=0.2, help="Drop-prob")
parser.add_option("--lr", dest="LR", default=1, help="How long to keep high LR")

#Set masking for amino acids within any 2 regions
# E.g. mask peptide sequence from 181-192 (--m1 181 --m2 192)
parser.add_option("--m1", "--mask1", dest="MASK1", default="", help="Set masking for any region in X")
parser.add_option("--m2", "--mask2", dest="MASK2", default="", help="Set masking for any region in X")
parser.add_option("--m3", "--mask3", dest="MASK3", default="", help="Set masking for any region in X")
parser.add_option("--m4", "--mask4", dest="MASK4", default="", help="Set masking for any region in X")

# Print network (layer) sizes
parser.add_option("-p", "--ps", dest="ps", default=False, help="Print network sizes")

#Remove later
# Load baseline?
parser.add_option("-l", "--load", dest="LOAD", default="", help="Load baseline trained models")

(options,args) = parser.parse_args()

MODE = int(options.MODE)
OUTDIR = str(options.OUTDIR)
COMMENT = str(options.COMMENT)
CSVFILE = str(options.CSVFILE)
PS = options.ps
SETS = int(options.SETS)

GPU = options.GPU

print("Input parameters:")
if options.RANDOM != False:
    RANDOM = True
    COMMENT += " RP"
    print("Random partition mode set")

if options.RP == True:
    COMMENT += " unitrp"
    print("Random project unit on")

if options.SETS != int(1):
    COMMENT += " S:" + str(SETS)
    print("Sets:", str(SETS))

if options.MASK1 != "" and options.MASK2 != "":
    MASK1 = int(options.MASK1)
    MASK2 = int(options.MASK2)
    COMMENT += " M:" + str(MASK1) + "_" + str(MASK2)
    print("Masking in region", str(MASK1) + str(MASK2))

if options.MASK3 != "" and options.MASK4 != "":
    MASK3 = int(options.MASK3)
    MASK4 = int(options.MASK4)
    COMMENT += " M2:" + str(MASK3) + "_" + str(MASK4)
    print("Masking in region", str(MASK3) + str(MASK4))

if options.DP != 0.2:
    COMMENT += " dp:" + str(options.DP)
    print("Drop prob set to", str(options.DP))

print("Comment:", COMMENT)
print("Model outdir:", OUTDIR)
print("CSV path:", CSVFILE)

os.makedirs(OUTDIR, exist_ok = True)

#############################
# Load data
#############################

print("Loading data by ordered partitions ...")
p0 = glob.glob("../data/train_data_new/*1p*")
p1 = glob.glob("../data/train_data_new/*2p*")
p2 = glob.glob("../data/train_data_new/*3p*")
p3 = glob.glob("../data/train_data_new/*4p*")
p4 = glob.glob("../data/train_data_new/*5p*")

print("Partition sizes:", len(p0), len(p1), len(p2), len(p3), len(p4))

X_p0, y_p0 = load_data(p0)
X_p1, y_p1 = load_data(p1)
X_p2, y_p2 = load_data(p2)
X_p3, y_p3 = load_data(p3)
X_p4, y_p4 = load_data(p4)

X_partitions = [X_p0, X_p1, X_p2, X_p3, X_p4]
y_partitions = [y_p0, y_p1, y_p2, y_p3, y_p4]

print("Done loading data")

#############################
# Model Start
#############################

print("Setting up model ...")

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
ps = True

#############################
# Run model
#############################

#data = [p0, p1, p2, p3, p4]
partitions = [0, 1, 2, 3, 4]

sz0 = 896
m1, m2 = generate_weights(32, sz0, 0, GPU)
batch_size = 32

print("dp:", options.DP)
drop_prob = 0.2
if options.DP != str(0.2):
    print("Setting drop prob to", options.DP)
    drop_prob = float(options.DP)
    print("New dp:", drop_prob)

### Cross-validation mode ###
if MODE == 1:

    for i in partitions:
        #valid = data[i]
        #train = [item for sublist in data[:i] + data[i+1:] for item in sublist]
        #test = valid

        train_str = str(partitions[:i] + partitions[i+1:]).strip('[]')
        print("\nRun", i, "/ 5 ...", "Test", i, "Train", train_str)

        #Load data
        #X, y, X_val, y_val, _, _ = data_generator(train, valid, test)
        #X, y = data[i]
        #sys.exit()
        #X, y = 
        #X = np.empty((0, 420, 142))
        #y = np.empty(0)
        X = None
        #y = []        
        for i2 in range(5):
            if i2 != i:
                #if i2 == 0 or (i2 == 1 and i == 0):
                if X is None:
                    X = X_partitions[i2]
                    y = y_partitions[i2]
                else:
                    X = np.concatenate((X, X_partitions[i2]))
                    y = np.concatenate((y, y_partitions[i2]))
        #X, y = np.delete(data, i)
        X_val, y_val = X_partitions[i], y_partitions[i]
        print(X.shape)
        print(y.shape)
        print(X_val.shape)
        X0, y0, X0_val, y0_val = X.copy(), y.copy(), X_val.copy(), y_val.copy()

        #Masking
        if options.MASK1 != "" and options.MASK2 != "":
            print(X.shape)
            X[:, MASK1:MASK2, 0:20] = np.zeros((X[:, MASK1:MASK2, 0:20]).shape)
            if PS: print("Masked sequence position (only AAs)", MASK1, MASK2, X.shape)

        if options.MASK3 != "" and options.MASK4 != "":
            print(X.shape)
            X[:, MASK3:MASK4, 0:20] = np.zeros((X[:, MASK3:MASK4, 0:20]).shape)
            if PS: print("Masked sequence position (only AAs)", MASK3, MASK4, X.shape)

        #Upsample
        X, y = upsample(X, y)

        X = to_torch_data(X,float,torch.float32)
        y = to_torch_data(y,int,torch.int64)
        X0_val = to_torch_data(X0_val,float,torch.float32)
        y0_val = to_torch_data(y0_val,int,torch.int64)

        #Create Tensor Dataset and FastAI databunch
        train_ds = tdatautils.TensorDataset(X, y)
        valid_ds0 = tdatautils.TensorDataset(X0_val, y0_val)
        dummy_ds = tdatautils.TensorDataset(X0_val[0:2], y0_val[0:2])
        my_data_bunch = DataBunch.create(train_ds, valid_ds0, bs=batch_size)

        #Initialize model
        if GPU:
            net = Model().cuda()
        else:
            net = Model()
        learn = Learner(my_data_bunch, net,
                             opt_func=torch.optim.Adam,
                             loss_func=criterion, metrics=[accuracy],
                             wd = 0.01)

        if options.LOAD != "":
            learn.load(baseline[i])
        if MODE == 1:
            cycles = SETS
            #epochs = 2*[1]+(SETS-2)*[1]
            epochs = [3, 3, 3, 3]
            #multiply high LR by options.LR
            LR = [1e-02, 1e-01, 1e-02, 1e-01, 5e-03, 5e-02, 1e-03, 1e-02] * int(options.LR) + (SETS-4)*[5e-03, 5e-02]
            filepath = simple_train(cycles = cycles, epochs = epochs, LR = LR, val_part = i, skip = False)

        #Stats
        learn = Learner(my_data_bunch,
                             net,
                             opt_func=torch.optim.Adam,
                             loss_func=criterion, metrics=[accuracy],
                             callback_fns=[partial(EarlyStoppingCallback, min_delta=0.01, patience=3)],
                             wd = 0.01)

        print("Model saved:\n", filepath)
        learn.load(filepath)
        stats_to_csv(comment = COMMENT, val_part = i, LR=LR, ds = DatasetType.Valid)

#############################
# Run model using nested-cross-val
#############################

if MODE == 2:
    print("Mode: 2. Nested-cross validation")
    data = [p0, p1, p2, p3, p4]
    for i in partitions:
        valid = data[i]

        for i2 in partitions:
            if i2 == i:
                continue
            test = data[i2]
            train_index = []
            for i3 in range(5):
                if i3 != i and i3 != i2:
                    train_index.append(i3)
            train = data[train_index]

            print(i, "/ 20", " ... Val", i, "Test", i2, "Train", train_index)

            #Load data
            Xt, yt, Xt_val, yt_val, Xt_test, yt_test = data_generator(train, valid, test)
            X0, y0, X0_val, y0_val, X0_test, y0_test = Xt.copy(), yt.copy(), Xt_val.copy(), yt_val.copy(), Xt_test.copy(), yt_test.copy()

            #Upsample
            Xp, yp = upsample(Xt, yt)
            Xp_val, yp_val = upsample(Xt_val, yt_val)
            Xp_test, yp_test = upsample(Xt_test, yt_test)

            X, X_val, X_test = map(lambda x: to_torch_data(x,float,torch.float32),(Xp, Xp_val, Xp_test))
            y, y_val, y_test = map(lambda x: to_torch_data(x,int,torch.int64),(yp, yp_val, yp_test))
            X0, X0_val, X0_test = map(lambda x: to_torch_data(x,float,torch.float32),(X0, X0_val, X0_test))
            y0, y0_val, y0_test = map(lambda x: to_torch_data(x,int,torch.int64),(y0, y0_val, y0_test))

            train_ds = tdatautils.TensorDataset(X, y)
            valid_ds0 = tdatautils.TensorDataset(X0_val, y0_val)
            test_ds0 = tdatautils.TensorDataset(X0_test, y0_test)

            #Run model
            my_data_bunch = DataBunch.create(train_ds, valid_ds0, test_ds0, bs=batch_size)
            if GPU:
                net = Model().cuda()
            else:
                net = Model()
            learn = Learner(my_data_bunch,
                            net,
                            opt_func=torch.optim.Adam,
                            loss_func=criterion, metrics=[accuracy],
                            callback_fns=[partial(EarlyStoppingCallback, min_delta=0.01, patience=3)],
                            wd = 0.01)

            #Stats
            filepath = simple_train(cycles = 4, epochs = 4*[3], LR = [1e-02, 1e-01, 1e-02, 1e-01, 1e-03, 5e-02, 1e-03, 5e-02], val_part = i2, test_part = i)

            print("Model saved:\n", filepath)
            learn.load(filepath)
            stats_to_csv(comment = "NCV_val 4cyc "+COMMENT, val_part = i2, test_part = i, ds = DatasetType.Valid)
            stats_to_csv(comment = "NCV_test 4cyc "+COMMENT, val_part = i2, test_part = i, ds = DatasetType.Test)


#############################
# Calculate nested cross-val performance from saved models
#############################
# Runs automatically if nested-cross val mode is set (MODE == 2)
if MODE == 2:

    print("2. Checking nested-cross val test sets")

    #Check right number of partitions
    saved_models = glob.glob(OUTDIR)
    print(len(saved_models) == 20, len(saved_models), "== 20")

    preds = []
    names = []
    targets = []

    for filename in saved_models:
        test_part = int(re.search('.*T(\d).*', filename).group(1))
        val_part = int(re.search('.*V(\d)_', filename).group(1))
        filepath = filename[:-4]

        print("Test", test_part, "val", val_part)
        print(filepath)

        #Set train and valid parts
        train = data[val_part]
        valid = data[test_part]

        #Load data
        X,y, X_val, y_val, _, _ = data_generator(train, valid, valid)
        X, X_val = map(lambda x: to_torch_data(x,float,torch.float32),(X, X_val))
        y, y_val = map(lambda x: to_torch_data(x,int,torch.int64),(y, y_val))
        train_ds = tdatautils.TensorDataset(X, y)
        valid_ds = tdatautils.TensorDataset(X_val, y_val)

        #Setup model
        my_data_bunch = DataBunch.create(train_ds, valid_ds, bs=batch_size)
        if GPU:
            net = Model().cuda()
        else:
            net = Model()
        learn = Learner(my_data_bunch,
                             net,
                             opt_func=torch.optim.Adam,
                             loss_func=criterion, metrics=[accuracy],
                             callback_fns=[partial(EarlyStoppingCallback, min_delta=0.01, patience=3)],
                             wd = 0.01)

        #Stats
        learn.load(filepath)

        #Extract predictions
        learn_preds = learn.get_preds()
        y_hat = learn_preds[0][:, 1]
        y_true = learn_preds[1]

        names.append("T" + str(test_part) + "V" + str(val_part))
        preds.append(y_hat)
        targets.append(y_true)

    #Save to file
    names = pd.DataFrame(names)
    preds = pd.DataFrame(preds)
    targets = pd.DataFrame(targets)

    pd.to_pickle(names, "names")
    pd.to_pickle(preds, "preds")
    pd.to_pickle(targets, "targets")

    print("Checking values from previously saved models")

    #Check predictions
    print("Loading predictions, names and targets data ...")

    #Load data
    names = pd.read_pickle("names")

    y_hat = pd.read_pickle("preds")
    y_hat = y_hat.astype(np.float64)

    y_true = pd.read_pickle("targets")

    #Averaging
    pp0 = np.average(y_hat[0:4], axis = 0)
    pp1 = np.average(y_hat[4:8], axis = 0)
    pp2 = np.average(y_hat[8:12], axis = 0)
    pp3 = np.average(y_hat[12:16], axis = 0)
    pp4 = np.average(y_hat[16:20], axis = 0)

    all_predictions = [pp0, pp1, pp2, pp3, pp4]
    test_sets = names.iloc[[0, 4, 8, 12, 16]].values

    #Calculate statistics
    for i, name in enumerate(test_sets):
        test_part = int(re.search('.*T(\d).*', str(name)).group(1))


        outputs = np.array(all_predictions[i])
        outputs = outputs[~np.isnan(outputs)].astype(np.float64)

        targets = np.array(y_true.iloc[i*4, :].dropna())
        targets = targets.astype(np.int64)
        stats_to_csv_nested(outputs, targets, val_part = test_part, comment = "1stNCV")

