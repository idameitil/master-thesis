import subprocess
import numpy as np

origin_file = open("/home/projects/ht3_aim/people/idamei/data/all_data_final_origin.csv", "r")
origin_dict = dict()
for line in origin_file:
    splitted_line = line.strip().split(",")
    entry_id = splitted_line[0]
    partition = splitted_line[4]
    origin = splitted_line[6]
    if origin == "tenX":
        origin = "tenx"
    elif origin == "swapped":
        origin = "swap"
    elif origin == "positive":
        origin = "pos"
    else:
        print("ERROR")
    origin_dict[entry_id] = {"partition":partition, "origin":origin}
origin_file.close()

old_array_folder = "/home/projects/ht3_aim/people/idamei/results/energy_output_arrays/"
new_array_folder = "/home/projects/ht3_aim/people/idamei/data/train_data/"
for partition in range(1,6):
    for binder in ["positives", "negatives"]:
        p = subprocess.Popen(["ls", f"{old_array_folder}{str(partition)}/{binder}/"],
                             stdout=subprocess.PIPE, universal_newlines=True)
        filenames = p.communicate()[0].split()
        if binder == "positives":
            binder = "pos"
        elif binder == "negatives":
            binder = "neg"
        for filename in filenames:
            entry_id = filename.replace(".npy", "")
            partition = origin_dict[entry_id]["partition"]
            origin = origin_dict[entry_id]["origin"]
            new_filename = f"{entry_id}_{partition}p_{binder}_{origin}.npy"

            old_array = np.load(f"{old_array_folder}{str(partition)}/{binder}/{filename}")
            new_array = old_array[:-4]

            np.save(new_array, new_array_folder+new_filename)