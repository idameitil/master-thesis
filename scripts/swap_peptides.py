import random
from pprint import pprint
from collections import Counter

# Make list of peptides
infile1 = open("/home/ida/master-thesis/data/positives.csv", "r")
peptides = [[],[],[],[],[]]
for line in infile1:
    if line.startswith("#"):
        continue
    splitted_line = line.split(",")
    peptide = splitted_line[2]
    partition = int(splitted_line[3])
    peptides[partition-1].append(peptide)
infile1.close()
random.shuffle(peptides[0])
random.shuffle(peptides[1])
random.shuffle(peptides[2])
random.shuffle(peptides[3])
random.shuffle(peptides[4])

outfile1 = open("/home/ida/master-thesis/results/peptide_counts_positives.txt", "w")
for partition in peptides:
    peptides_in_part = dict()
    for peptide in partition:
        if peptide in peptides_in_part:
            peptides_in_part[peptide] += 1
        else:
            peptides_in_part[peptide] = 1
    pprint(peptides_in_part, outfile1)
outfile1.close()

infile2 = open("/home/ida/master-thesis/data/positives.csv", "r")
outfile2 = open("/home/ida/master-thesis/results/swap_negatives.txt", "w")
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

    for i in range(len(peptides[int(partition)-1])):
        random_peptide = peptides[int(partition)-1][i]
        if random_peptide != peptide:
            break
        else:
            pass
    del peptides[int(partition)-1][i]

    outfile2.write(",".join([CDR3a, CDR3b, peptide, random_peptide, partition, str(0), v_gene_alpha, j_gene_alpha, v_gene_beta, \
                            j_gene_beta, CDR3a_full, CDR3b_full]))
infile2.close()
outfile2.close()

# Make list of peptides
infile3 = open("/home/ida/master-thesis/data/tcr_data/negatives_10x_ab95.csv", "r")
peptides = [[],[],[],[],[]]
for line in infile3:
    if line.startswith("#"):
        continue
    splitted_line = line.split(",")
    peptide = splitted_line[2]
    partition = int(splitted_line[3])
    peptides[partition-1].append(peptide)
infile3.close()
random.shuffle(peptides[0])
random.shuffle(peptides[1])
random.shuffle(peptides[2])
random.shuffle(peptides[3])
random.shuffle(peptides[4])

outfile3 = open("/home/ida/master-thesis/results/peptide_counts_negatives.txt", "w")
for partition in peptides:
    peptides_in_part = dict()
    for peptide in partition:
        if peptide in peptides_in_part:
            peptides_in_part[peptide] += 1
        else:
            peptides_in_part[peptide] = 1
    pprint(peptides_in_part, outfile3)

infile3.close()
outfile3.close()