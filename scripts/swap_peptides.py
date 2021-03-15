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

infile2 = open("/home/ida/master-thesis/data/positives.csv", "r")
outfile2 = open("/home/ida/master-thesis/results/swap_negatives.txt", "w")
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

# Make list of peptides
infile3 = open("/home/ida/master-thesis/data/tcr_data/negatives_10x_ab95.csv", "r")
peptides = [[],[],[],[],[]]
unique_peptides_negative = []
for line in infile3:
    if line.startswith("#"):
        continue
    splitted_line = line.split(",")
    peptide = splitted_line[2]
    partition = int(splitted_line[3])
    peptides[partition-1].append(peptide)
    if not peptide in unique_peptides_negative:
        unique_peptides_negative.append(peptide)
infile3.close()
random.shuffle(peptides[0])
random.shuffle(peptides[1])
random.shuffle(peptides[2])
random.shuffle(peptides[3])
random.shuffle(peptides[4])

# Count negative peptides
peptide_counts_negatives = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
frequencies_negatives = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
for partition in range(1,6):
    peptides_in_part = dict()
    peptides_in_partition = peptides[partition-1]
    peptide_counts_negatives[partition] = dict((el, 0) for el in unique_peptides)
    for peptide in peptides_in_partition:
        peptide_counts_negatives[partition][peptide] += 1
    total_count_peptides = sum(peptide_counts_negatives[partition].values())
    frequencies = {k: v / total_count_peptides for k, v in peptide_counts_negatives[partition].items()}
    frequencies_negatives[partition]=frequencies
outfile3 = open("/home/ida/master-thesis/results/peptide_counts_negatives.txt", "w")
pprint(peptide_counts_negatives, outfile3)
pprint(frequencies_negatives, outfile3)
outfile3.close()

# Find all unique peptides from negatives and positives
unique_peptides_total = set(unique_peptides_negative + unique_peptides_positive)

# Add 10X negatives and swapped negatives
peptide_counts_total_negatives = {1:dict((el, 0) for el in unique_peptides), 2:dict((el, 0) for el in unique_peptides), 3:dict((el, 0) for el in unique_peptides), 4:dict((el, 0) for el in unique_peptides), 5:dict((el, 0) for el in unique_peptides)}
for partition in range(1,6):
    for peptide in unique_peptides:
        peptide_counts_total_negatives[partition][peptide] = peptide_counts_negatives[partition][peptide] + peptide_counts_swapped[partition][peptide]
frequencies_total_negatives= {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
for partition in range(5):
    total_count_peptides = sum(peptide_counts_total_negatives[partition+1].values())
    frequencies = {k: v / total_count_peptides for k, v in peptide_counts_total_negatives[partition+1].items()}
    frequencies_total_negatives[partition+1] = frequencies

for partition in range(1,6):
    labels = unique_peptides_total
    positives = [frequencies_positives[partition][k] for k in labels]
    negatives = [frequencies_negatives[partition][k] for k in labels]
    negatives_total = [frequencies_total_negatives[partition][k] for k in labels]
    x = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, positives, width, label='Positives')
    rects2 = ax.bar(x, negatives, width, label='Negatives 10X')
    rects3 = ax.bar(x + width, negatives_total, width, label='10X plus swapped negatives')

    ax.set_title("Partition " + str(partition))
    ax.set_ylabel('Frequency')
    #ax.set_xticks(x, rotation='vertical')
    #ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.xticks(x, labels, rotation='vertical')
    # Pad margins so that markers don't get clipped by the axes
    #plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.23)
    fig.savefig(fname = "/home/ida/master-thesis/results/frequency_barplot_P{}.png".format(partition))
