
__requires__ = 'bcr-models==0.0.0'
import bcr_models as bcr
import re
import sys
from Bio import SeqIO

# Usage: "python TCR_gene2seq.py <infile> <outfile>"

##################################################
###                Functions                   ###
##################################################

def find_sequence(gene_name):
    """Retrieves acti, t, f, g, a from gene name and looks up in genes dict.
    Returns sequence from the given gene.
    Global: genes_dict"""
    # Retrieve acti, t, f, g and a
    rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', gene_name)
    if (rege):
        actg = "1"
        acta = "1"
        acti = rege.group(1)
        actt = rege.group(2)
        actf = rege.group(3)
        more = rege.group(4)
        if more:
            rege2 = re.match(r'\-0*(\d+)', more)
            if (rege2):
                actg = rege2.group(1)
            rege3 = re.match(r'\*0*(\d+)', more)
            if (rege3):
                acta = rege3.group(1)
        # Look up in dictionary
        try:
            sequence = genes_dict[(acti, actt, actf, actg, acta)]
        except:
            print("Error: Could not find gene in database: " + gene_name, acti, actt, actf,
                  actg, acta)
    else:
        print("Error: Could not recognize gene name: {}".format(gene_name))
    return sequence

def align_sequence(cdr3, v_seq, j_seq):
    """Combines CDR3, V and J sequence"""
    aligned_seq = ''

    ig_chain2 = bcr.IgChain(v_seq, template_db=template_db, pdb_db=pdb_db)
    ig_chain3 = bcr.IgChain(j_seq, template_db=template_db, pdb_db=pdb_db)

    # Look for "C" within last 6 AAs in V gene
    rege1 = re.match(r'(.+C)[^C]{0,5}', ig_chain2.sequence)
    if rege1:
        chain2 = rege1.group(1)
        # Look for "F" or "W" within first 11 AAs followed by "G"
        rege2 = re.match(r'.{0,11}([FW]G.*)', ig_chain3.sequence)
        if rege2:
            chain3 = rege2.group(1)
            # Combine and do hmmsearch
            finalseq = chain2 + cdr3 + chain3
            final_ig = bcr.IgChain(finalseq, template_db=template_db, pdb_db=pdb_db)
            try:
                final_ig.hmmsearch(*hmms)
                aligned_seq = final_ig.aligned_seq
            except Exception:
                print("Error: hmmsearch failed for: " + finalseq)
        else:
            print("Error: Could not find 'F/W' followed by 'G' in J gene: " + \
                  j_gene_alpha + " " + ig_chain3.sequence)
    else:
        print("Error: Could not find a 'C' within last six AAs in V gene: " + \
              v_gene_alpha + " " + ig_chain2.sequence)
    return aligned_seq

##################################################
###      Make dictionary of TCR genes          ###
##################################################

gene_sequences = SeqIO.parse(open('TCR_genes.fasta'),'fasta')
genes_dict = {}
for entry in gene_sequences:
    name, sequence = entry.id, str(entry.seq)
    rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)', name)
    if (rege):
        actg = "1"
        acta = "1"
        acti = rege.group(1)
        actt = rege.group(2)
        actf = rege.group(3)
        more = rege.group(4)
        rege2 = re.search(r'\-0*(\d+)', more)
        if (rege2):
            actg = rege2.group(1)
        rege2 = re.search(r'\*0*(\d+)', more)
        if (rege2):
            acta = rege2.group(1)
        genes_dict[(acti, actt, actf, actg, acta)] = sequence
    else:
        print("Error: Could not understand gene name in database file:" + name)

###############################################
### Retrieve full sequences for input file  ###
###############################################

hmms = bcr.db.builtin_hmms()
template_db = bcr.db.BuiltinTemplateDatabase()
pdb_db = bcr.db.BuiltinPDBDatabase()

input_filename = sys.argv[1]   #ID,CDR3a,CDR3b,v_gene_alpha,j_gene_alpha,v_gene_beta,j_gene_beta
infile = open(input_filename, "r")
output_filename = sys.argv[2]
outfile = open(output_filename, "w")
outfile.write("#ID,TCRa_sequence,TCRb_sequence\n")

for line in infile:
    if line.startswith("#"):
        continue
    ID, CDR3a, CDR3b, v_gene_alpha, j_gene_alpha, v_gene_beta, j_gene_beta = line.strip().split(",")
    # Get alpha sequence
    v_seq_alpha = find_sequence(v_gene_alpha)
    j_seq_alpha = find_sequence(j_gene_alpha)
    aligned_seq_alpha = align_sequence(CDR3a, v_seq_alpha, j_seq_alpha)

    # Get beta sequence
    v_seq_beta = find_sequence(v_gene_beta)
    j_seq_beta = find_sequence(j_gene_beta)
    aligned_seq_beta = align_sequence(CDR3b, v_seq_beta, j_seq_beta)

    # Write to outfile
    if aligned_seq_alpha != '' and aligned_seq_beta != '':
        TCRa_seq = aligned_seq_alpha.replace("-", "")
        TCRb_seq = aligned_seq_beta.replace("-", "")
        outfile.write('{},{},{}\n'.format(ID,TCRa_seq,TCRb_seq))
