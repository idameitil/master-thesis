import glob

data_dir = "/home/ida/master-thesis/data/train_data/*"
outfile = open("/home/ida/master-thesis/data/IDs_in_train_set.txt", "w")

for filename in glob.glob(data_dir):
    ID = filename.replace("/home/ida/master-thesis/data/train_data/", "").split("_")[0]
    outfile.write(ID + "\n")

outfile.close()