
unique_peptides = ['CLGGLLTMV', 'KLQCVDLHV', 'FLYALALLL', 'ILKEPVHGV', 'SLFNTVATL', 'LLFGYPVYV', 'SLLMWITQV', 'MLDLQPETT', 'KVLEYVIKV', 'KTWGQYWQV', 'YLLEMLWRL', 'RMFPNAPYL', 'KVAELVHFL', 'NLVPMVATV', 'GILGFVFTL', 'IMDQVPFSV', 'RTLNAWVKV', 'GLCTLVAML']

positive_file = open("/home/ida/master-thesis/data/positives.csv")
negative_file = open("/home/ida/master-thesis/data/tcr_data/negatives_10x_ab95.csv")
swap_file = open("/home/ida/master-thesis/data/swap_negatives.txt", "r")

# Count positive peptides
peptide_counts_positive = {1:dict((el, 0) for el in unique_peptides), 2:dict((el, 0) for el in unique_peptides), 3:dict((el, 0) for el in unique_peptides), 4:dict((el, 0) for el in unique_peptides), 5:dict((el, 0) for el in unique_peptides)}
for line in positive_file:
    if line.startswith("#"):
        continue
    splitted_line = line.strip().split(",")
    partition = int(splitted_line[3])
    peptide = splitted_line[2]
    peptide_counts_positive[partition][peptide] += 1

# Count negative peptides
peptide_counts_negative = {1:dict((el, 0) for el in unique_peptides), 2:dict((el, 0) for el in unique_peptides), 3:dict((el, 0) for el in unique_peptides), 4:dict((el, 0) for el in unique_peptides), 5:dict((el, 0) for el in unique_peptides)}
for line in negative_file:
    if line.startswith("#"):
        continue
    splitted_line = line.strip().split(",")
    partition = int(splitted_line[3])
    peptide = splitted_line[2]
    peptide_counts_negative[partition][peptide] += 1

# Count swapped peptides
peptide_counts_swapped = {1:dict((el, 0) for el in unique_peptides), 2:dict((el, 0) for el in unique_peptides), 3:dict((el, 0) for el in unique_peptides), 4:dict((el, 0) for el in unique_peptides), 5:dict((el, 0) for el in unique_peptides)}
for line in swap_file:
    if line.startswith("#"):
        continue
    splitted_line = line.strip().split(",")
    partition = int(splitted_line[3])
    peptide = splitted_line[2]
    peptide_counts_swapped[partition][peptide] += 1

# Add 10X and swapped
peptide_counts_total_negatives = {1:dict((el, 0) for el in unique_peptides), 2:dict((el, 0) for el in unique_peptides), 3:dict((el, 0) for el in unique_peptides), 4:dict((el, 0) for el in unique_peptides), 5:dict((el, 0) for el in unique_peptides)}
for partition in range(1,6):
    for peptide in unique_peptides:
        peptide_counts_total_negatives[partition][peptide] = peptide_counts_negative[partition][peptide] + peptide_counts_swapped[partition][peptide]

# Calculate positive/negative ratio
ratio_positive_negative = {1:dict((el, 0) for el in unique_peptides), 2:dict((el, 0) for el in unique_peptides), 3:dict((el, 0) for el in unique_peptides), 4:dict((el, 0) for el in unique_peptides), 5:dict((el, 0) for el in unique_peptides)}
for partition in range(1,6):
    for peptide in unique_peptides:
        try:
            ratio_positive_negative[partition][peptide] = peptide_counts_total_negatives[partition][peptide] / peptide_counts_positive[partition][peptide]
        except:
            pass
print(ratio_positive_negative)