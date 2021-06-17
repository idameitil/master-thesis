from pprint import pprint

vdjdb_file = open("data/VDJdb_minscore0_2021-02-11.tsv", "r")
TRB_cdrs = dict()
TRA_cdrs = dict()
wrong_start = 0
wrong_end = 0
for line in vdjdb_file:
    splitted_line = line.split("\t")
    gene = splitted_line[1]
    cdr3_full = splitted_line[2]
    if cdr3_full.startswith("C"):
        cdr3_trimmed = cdr3_full[1:]
    else:
        wrong_start += 1
        continue
    if cdr3_full.endswith("F") or cdr3_full.endswith("W"):
        cdr3_trimmed = cdr3_trimmed[:-1]
    else:
        wrong_end += 1
        continue
    V = splitted_line[3]
    J = splitted_line[4]
    if gene == "TRB":
        TRB_cdrs[cdr3_trimmed] = [V, J, cdr3_full]
    elif gene == "TRA":
        TRA_cdrs[cdr3_trimmed] = [V, J, cdr3_full]
vdjdb_file.close()

train_file = open("data/NetTCR2_data/train_ab95_5x_umi0.csv", "r")
outfile = open("positives.csv", "w")
outfile.write("#CDR3a,CDR3b,peptide,partition,binder,v_gene_alpha,j_gene_alpha,v_gene_beta,j_gene_beta,cdr3a_full,cdr3b_full\n")

count_failed = 0
count_succeded = 0
failed_a = 0
failed_b = 0
count_single = 0
count_multiple = 0
count_missing_genes = 0
for line in train_file:
    line = line.strip()
    if line.startswith("#"):
        continue
    splitted_line = line.split(",")
    if splitted_line[4] == "0":    # only positives
        continue
    # Save cdr3a
    cdr3a = splitted_line[0]
    # Save cdr3b
    cdr3b = splitted_line[1]
    # Find beta genes
    beta_succeded = False
    try:
        v_gene_beta, j_gene_beta, cdr3_full_beta = TRB_cdrs[cdr3b]
        beta_succeded = True
    except KeyError:
        pass
    # If previous fails, remove W if there
    if not beta_succeded:
        if cdr3b.endswith("W") or cdr3b.endswith("F"):
            cdr3b = cdr3b[:-1]
        if cdr3b.startswith("C"):
            cdr3b = cdr3b[1:]
        try:
            v_gene_beta, j_gene_beta, cdr3_full_beta = TRB_cdrs[cdr3b]
            beta_succeded = True
        except KeyError:
            pass
    # Find alpha genes
    alpha_succeded = False
    try:
        v_gene_alpha, j_gene_alpha, cdr3_full_alpha = TRA_cdrs[cdr3a]
        alpha_succeded = True
    except KeyError:
        pass
    # If previous fails, remove W if there
    if not alpha_succeded:
        if cdr3a.endswith("W") or cdr3a.endswith("F"):
            cdr3a = cdr3a[:-1]
        if cdr3a.startswith("C"):
            cdr3a = cdr3a[1:]
        try:
            v_gene_alpha, j_gene_alpha, cdr3_full_alpha = TRA_cdrs[cdr3a]
            alpha_succeded = True
        except KeyError:
            pass
    if alpha_succeded and beta_succeded:
        if v_gene_alpha == "" or j_gene_alpha == "" or v_gene_beta == "" or j_gene_beta == "":
            count_failed += 1
            count_missing_genes += 1
            print("One or more genes missing in VDJdb for cdrs. cdra: {} cdrb: {}".format(cdr3a, cdr3b))
        else:
            outfile.write(line + "," + v_gene_alpha + "," + j_gene_alpha + "," + v_gene_beta + \
                      "," + j_gene_beta + "," + cdr3_full_alpha + "," + cdr3_full_beta + "\n")
            count_succeded += 1
    else:
        count_failed += 1
        if not alpha_succeded and not beta_succeded:
            pass
            #print("both failed. cdra: {} cdrb: {}".format(cdr3a, cdr3b))
        else:
            if not alpha_succeded:
                pass
                print("alpha failed: {}".format(cdr3a))
            if not beta_succeded:
                pass
                print("beta failed: {}".format(cdr3b))

print("VDJdb wrong start: {}. Wrong end: {}.". format(wrong_start, wrong_end))
#print("single: " + str(count_single) + ". Multiple: " + str(count_multiple) + ". Failed: " + str(count_failed))
print("Succeded: " + str(count_succeded) + ". Failed: " + str(count_failed))
print("Failed due to missing genes: {}".format(count_missing_genes))
#print("failed a: " + str(failed_a))
#print("failed b: " + str(failed_b))
train_file.close()
outfile.close()
