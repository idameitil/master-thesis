
__requires__ = 'bcr-models==0.0.0'
import re
import sys
import os
import pandas
import bcr_models as bcr
import bcr_models.utils
import bcr_models.canonical_structures

from pprint import pprint
import re
import csv

from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser(description='Transform TCR CDR+germ lines to complete sequences')
parser.add_argument('filein',  type=str, help='the csv-format file from VDJ_DB the following data: sequence ID, CDR3, V gene, J gene')

args = parser.parse_args()
filein= args.filein

data=pandas.read_csv(filein,   sep=",")

genes={}
gene_sequences = SeqIO.parse(open('/home/ida/master-thesis/data/TCR_genes.fasta'),'fasta')

for fasta in gene_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    rege=re.match(r'TR(.)(.)0*(\d+)([^\|]+)', name)
    if (rege):
        actg="1"
        acta="1"
        acti=rege.group(1)
        actt=rege.group(2)
        actf=rege.group(3)
        more=rege.group(4)
        rege2=re.search(r'\-0*(\d+)', more)
        if (rege2):
            actg=rege2.group(1)
        rege2=re.search(r'\*0*(\d+)', more)
        if (rege2):
            acta=rege2.group(1)
        genes[(acti,actt,actf,actg,acta)]=sequence
    else:
        print("Error. Could not understand name in database file:" + name)

hmms = bcr.db.builtin_hmms()
template_db = bcr.db.BuiltinTemplateDatabase()
pdb_db = bcr.db.BuiltinPDBDatabase()
#csdb = bcr.db.BuiltinCsDatabase()
sample_class={}
sample_memory={}

count_success = 0
error_1, error_2, error_3 = 0, 0, 0

with open( filein, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        vseq=''
        jseq=''
        # Find V sequence
        rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', rows[2])
        if (rege):
            actg_v="1"
            acta_v="1"
            acti_v=rege.group(1)
            actt_v=rege.group(2)
            actf_v=rege.group(3)
            more=rege.group(4)
            if more:
                rege2=re.match(r'\-0*(\d+)', more)
                if (rege2):
                    actg_v=rege2.group(1)
            if more:
                rege2=re.match(r'\*0*(\d+)', more)
                if (rege2):
                    acta_v=rege2.group(1)
            try:
                vseq=genes[(acti_v,actt_v,actf_v,actg_v,acta_v)]
            except:
                print("Error. Could not find V gene in database: " + rows[2], str(acti_v),str(actt_v),str(actf_v),str(actg_v),str(acta_v))
                continue
        else:
            print("Error: Could not recognize V gene: {}".format(rows[2]))
            continue

        # Find J sequence
        rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', rows[3])
        if (rege):
            actg_j="1"
            acta_j="1"
            acti_j=rege.group(1)
            actt_j=rege.group(2)
            actf_j=rege.group(3)
            more=rege.group(4)
            if more:
                rege2=re.match(r'\-0*(\d+)', more)
                if (rege2):
                    actg_j=rege2.group(1)
            if more:
                rege2=re.match(r'\*0*(\d+)', more)
                if (rege2):
                    acta_j=rege2.group(1)
            try:
                jseq=genes[(acti_j,actt_j,actf_j,actg_j,acta_j)]
            except:
                print("Error. Could not find J gene in database: " + rows[3] + acti_j,actt_j,actf_j,actg_j,acta_j)
                continue
        else:
            print("Error: Could not recognize J gene: {}".format(rows[3]))
            continue

        ig_chain1 =bcr.IgChain(rows[1], template_db=template_db, pdb_db=pdb_db)
        ig_chain2 =bcr.IgChain(vseq, template_db=template_db, pdb_db=pdb_db)
        ig_chain3 =bcr.IgChain(jseq, template_db=template_db, pdb_db=pdb_db)

        rege1=re.match(r'(.+C)[^C]{0,5}', ig_chain2.sequence)
        if rege1:
            chain2=rege1.group(1)
            rege2=re.match(r'.{0,11}([FW]G.*)', ig_chain3.sequence)
            if rege2:
                chain3=rege2.group(1)
                finalseq=chain2+rows[1]+chain3
                final_ig =bcr.IgChain(finalseq, template_db=template_db, pdb_db=pdb_db)
                try:
                    final_ig.hmmsearch(*hmms)
                    aligned_seq=final_ig.aligned_seq
                    print(aligned_seq)
                    count_success += 1
                except Exception:
                    print("Error. hmmsearch failed for: " + finalseq)
                    error_1 += 1
            else:
                print("Error. Could not find 'F/W' followed by G in J gene: " +  \
                      acti_j + actt_j + actf_j + actg_j + acta_j + " " + ig_chain3.sequence)
                error_2 += 1
        else:
            print("Error. Could not find a 'C' within last six AAs in V gene: " + \
                  acti_v + actt_v + actf_v + actg_v + acta_v + " " + ig_chain2.sequence)
            error_3 += 1
print("Success: " + str(count_success))
print("Error hmmsearch failed: " + str(error_1))
print("Error F/W in J gene: " + str(error_2))
print("Error C in V gene: " + str(error_3))