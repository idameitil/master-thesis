import numpy as np
import subprocess
import os
from Bio import SeqIO
from joblib import Parallel, delayed
import multiprocessing as mp
from datetime import datetime

def run_tcrpmhcmodels(fasta_path)
    try:
        tcrpmhc_models_command = "tcrpmhc_models -t {}".format(fasta_path)
        subprocess.run(tcrpmhc_models_command.split(),
                       stdout=subprocess.PIPE, universal_newlines=True)
    except Error as err:
        print("Error: " + err)

fasta_directory = "/home/people/idamei/fastas"

fastas = subprocess.run("ls -d $PWD/*".split())

pool = mp.pool(40)

pool.map(run_tcrpmhcmodels(), [fasta for fasta in fastas])

pool.close()