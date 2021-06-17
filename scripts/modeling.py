import subprocess
import os
import sys
import multiprocessing as mp
import time

def run_tcrpmhcmodels(entry_id, partition, binder):
    """global fasta_directory"""
    entry_id, partition, binder
    try:
        # Make output directory
        model_path = "/home/projects/ht3_aim/people/idamei/results/full_models/" + entry_id + "_model"
        subprocess.run(["mkdir", model_path])
        # Run tcrpmhcmodels
        tcrpmhc_models_command = "tcrpmhc_models -t {} -p {}".format(fasta_directory + entry_id + ".fsa", model_path)
        subprocess.run(tcrpmhc_models_command.split(),
                       stdout=subprocess.PIPE, universal_newlines=True, cwd=model_path)
        # Put pdb in pdb directory and remove output directory
        if binder == "0":
            binder = "negatives"
        elif binder == "1":
            binder = "positives"
        subprocess.run(["cp", model_path + "/model_TCR-pMHC.pdb", "/home/projects/ht3_aim/people/idamei/results/models/" + partition + "/" + binder + "/" + entry_id + "_model.pdb"])
    except Exception as err:
        print("Error: Could not model " + entry_id)
        print(err)

### MAIN ###

# Read MHC
infile = open("/home/projects/ht3_aim/people/idamei/data/HLA-02-01.fasta", "r")
mhc_seq = ""
for line in infile:
    if line.startswith(">"):
        continue
    mhc_seq += line.strip()
infile.close()

# Write fastas
fasta_directory = "/home/projects/ht3_aim/people/idamei/data/fastas/"
infile = open("/home/projects/ht3_aim/people/idamei/data/all_data_final.csv", "r")
entries = []
for line in infile:
    if line.startswith("#"):
        continue
    entry_id, tcra_seq, tcrb_seq, peptide_seq, partition, binder = line.strip().split(",")
    # Write fasta
    fasta = fasta_directory + entry_id + ".fsa"
    outfile = open(fasta, "w")
    outfile.write(">M\n{}\n>P\n{}\n>A\n{}\n>B\n{}\n".format(mhc_seq, peptide_seq, tcra_seq, tcrb_seq))
    outfile.close()
    entries.append([entry_id, partition, binder])
infile.close()

pool = mp.Pool(40)

part1 = entries[0:2000]
part2 = entries[2000:4000]
part3 = entries[4000:6000]
part4 = entries[6000:8000]
part5 = entries[8000:10000]
part6 = entries[10000:]
pool.starmap(run_tcrpmhcmodels, [entry for entry in part6])

pool.close()

