import random
from pprint import pprint
import copy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

unique_peptides = ['CLGGLLTMV', 'KLQCVDLHV', 'FLYALALLL', 'ILKEPVHGV', 'SLFNTVATL', 'LLFGYPVYV', 'SLLMWITQV', 'MLDLQPETT', 'KVLEYVIKV', 'KTWGQYWQV', 'YLLEMLWRL', 'RMFPNAPYL', 'KVAELVHFL', 'NLVPMVATV', 'GILGFVFTL', 'IMDQVPFSV', 'RTLNAWVKV', 'GLCTLVAML']

# Make list of peptides
infile1 = open("/home/ida/master-thesis/data/positives.csv", "r")
peptides = [[],[],[],[],[]]
unique_peptides_positive = []
for line in infile1:
    if line.startswith("#"):
        continue
    splitted_line = line.split(",")
    peptide = splitted_line[2]
    partition = int(splitted_line[3])
    peptides[partition-1].append(peptide)
    if not peptide in unique_peptides_positive:
        unique_peptides_positive.append(peptide)
infile1.close()
random.shuffle(peptides[0])
random.shuffle(peptides[1])
random.shuffle(peptides[2])
random.shuffle(peptides[3])
random.shuffle(peptides[4])
peptides_copy = copy.deepcopy(peptides)

# Count peptides in positive dataset
outfile1 = open("/home/ida/master-thesis/results/peptide_counts_positives.txt", "w")
peptide_counts_positives = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
frequencies_positives = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
for partition in range(1,6):
    peptides_in_part = dict()
    peptides_in_partition = peptides[partition-1]
    peptide_counts_positives[partition] = dict((el, 0) for el in unique_peptides)
    for peptide in peptides_in_partition:
        peptide_counts_positives[partition][peptide] += 1
    total_count_peptides = sum(peptide_counts_positives[partition].values())
    frequencies = {k: v / total_count_peptides for k, v in peptide_counts_positives[partition].items()}
    frequencies_positives[partition]=frequencies
pprint(peptide_counts_positives, outfile1)
pprint(frequencies_positives, outfile1)
outfile1.close()

# Make swapped negatives
infile2 = open("data/positives.csv", "r")
outfile2 = open("swap_negatives.txt", "w")
peptide_counts_swapped = {1:dict((el, 0) for el in unique_peptides), 2:dict((el, 0) for el in unique_peptides), 3:dict((el, 0) for el in unique_peptides), 4:dict((el, 0) for el in unique_peptides), 5:dict((el, 0) for el in unique_peptides)}
for line in infile2:
    if line.startswith("#"):
        continue
    splitted_line = line.split(",")
    CDR3a = splitted_line[0]
    CDR3b = splitted_line[1]
    peptide = splitted_line[2]
    partition = splitted_line[3]
    v_gene_alpha = splitted_line[5]
    j_gene_alpha = splitted_line[6]
    v_gene_beta = splitted_line[7]
    j_gene_beta = splitted_line[8]
    CDR3a_full = splitted_line[9]
    CDR3b_full = splitted_line[10]
    # Choose random peptide that is not equal to the peptide
    hit = False
    for i in range(len(peptides[int(partition)-1])):
        random_peptide = peptides[int(partition)-1][i]
        if random_peptide != peptide:
            hit = True
            del peptides[int(partition) - 1][i]
            break
    # If there are no more peptides left, take from the copy
    if not hit:
        while True:
            random_peptide = random.choice(peptides_copy[int(partition)-1])
            if random_peptide != peptide:
                hit = True
                break
    if not hit:
        print("ERROR")
    outfile2.write(",".join([CDR3a, CDR3b, random_peptide, partition, str(0), v_gene_alpha, j_gene_alpha, v_gene_beta, \
                            j_gene_beta+"\n"]))
    # Count peptides
    peptide_counts_swapped[int(partition)][random_peptide] += 1
frequencies_swapped = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
for partition in range(5):
    total_count_peptides = sum(peptide_counts_swapped[partition+1].values())
    frequencies = {k: v / total_count_peptides for k, v in peptide_counts_swapped[partition+1].items()}
    frequencies_swapped[partition+1] = frequencies
infile2.close()
outfile2.close()
