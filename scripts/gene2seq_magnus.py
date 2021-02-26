
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

parser = argparse.ArgumentParser(description='Transorm TCR CDR+germ lines to complete sequences')
parser.add_argument('filein',  type=str, help='the csv-format file from VDJ_DB the following data: sequence ID, CDR3, V gene, J gene')

args = parser.parse_args()
filein= args.filein

data=pandas.read_csv(filein,   sep="\t")

genes={}
gene_sequences = SeqIO.parse(open('TCR_genes.fasta'),'fasta')

for fasta in gene_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    # print("Name " + name)
    rege=re.match(r'TR(.)(.)0*(\d+)([^\|]+)', name)
    if (rege):
        actg="1"
        acta="1"
        acti=rege.group(1)
        actt=rege.group(2)
        actf=rege.group(3)
        more=rege.group(4)
        rege2=re.match(r'\-0*(\d+)', more)
        if (rege2):
            actg=rege2.group(1)
        rege2=re.match(r'\*0*(\d+)', more)
        if (rege2):
            acta=rege2.group(1)
        # print("Sequence : " + sequence)
        genes[(acti,actt,actf,actg,acta)]=sequence
        #print("gene " + name + " : " + str(acti)+str(actt)+str(actf)+str(actg)+str(acta))
#pprint(genes)
hmms = bcr.db.builtin_hmms()
template_db = bcr.db.BuiltinTemplateDatabase()
pdb_db = bcr.db.BuiltinPDBDatabase()
#csdb = bcr.db.BuiltinCsDatabase()
sample_class={}
sample_memory={}


with open( filein, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        vseq=''
        jseq=''
        rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', rows[2])
        #rege=re.match(r'TCR(.)(.)0*(\d+)([^\|]+)', rows[2])
        if (rege):
            actg="1"
            acta="1"
            acti=rege.group(1)
            actt=rege.group(2)
            actf=rege.group(3)
            more=rege.group(4)
            if more:
                rege2=re.match(r'\-0*(\d+)', more)
                if (rege2):
                    actg=rege2.group(1)
            if more:
                rege2=re.match(r'\*0*(\d+)', more)
                if (rege2):
                    acta=rege2.group(1)
            try:
                vseq=genes[(acti,actt,actf,actg,acta)]
                print(rows[2], acti,actt,actf,actg,acta)
                # print("Vseq = "+ vseq)
            except:
                print("Vseq nono " + rows[2], str(acti),str(actt),str(actf),str(actg),str(acta))
        else:
            #pass
            print("Could not recognize V gene: {}".format(rows[2]))

        rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', rows[3])
        #rege=re.match(r'TCR(.)(.)0*(\d+)([^\|]+)', rows[3])
        if (rege):
            actg="1"
            acta="1"
            acti=rege.group(1)
            actt=rege.group(2)
            actf=rege.group(3)
            more=rege.group(4)
            if more:
                rege2=re.match(r'\-0*(\d+)', more)
                if (rege2):
                    actg=rege2.group(1)
            if more:
                rege2=re.match(r'\*0*(\d+)', more)
                if (rege2):
                    acta=rege2.group(1)
            try:
                jseq=genes[(acti,actt,actf,actg,acta)]
                print(rows[3], acti,actt,actf,actg,acta)
                # print("Jseq = "+ jseq)
            except:
                pass
                print("Jseq nono " + rows[3] + acti,actt,actf,actg,acta)
        else:
            pass
            print("Could not recognize J gene: {}".format(rows[3]))
        #
        # if re.match(r'^[ATGCU]+$', rows[1], flags=re.IGNORECASE):
        #     seq =rows[1].lower()
        #     chains = {}
        #     #Make six different IgChains
        #
        #     # print("Checking frame "+str(reading_frame) )
        #     # print('tmp seq is '+str(tmp_seq))
        #     ig_chain =bcr.IgChain(seq, template_db=template_db, pdb_db=pdb_db)
        #     #Score
        #     try:
        #         ig_chain.hmmsearch(*hmms)
        #     except:
        #         continue
        #     ig_chain1=ig_chain
        #
        # else:
        #     tmp_seq = rows[1]
        #     #print('tmp seq is '+str(tmp_seq))
        #     ig_chain1 =bcr.IgChain(tmp_seq, template_db=template_db, pdb_db=pdb_db)
        #     #Score
        #
        # tmp_seq = vseq
        # #print('tmp seq is '+str(tmp_seq))
        # ig_chain2 =bcr.IgChain(tmp_seq, template_db=template_db, pdb_db=pdb_db)
        # #Score
        #
        # tmp_seq = jseq
        # # print('tmp seq is '+str(tmp_seq))
        # ig_chain3 =bcr.IgChain(tmp_seq, template_db=template_db, pdb_db=pdb_db)
        # #Score
        #
        # finalseq=ig_chain1.sequence
        # rege1=re.match(r'(.+)C[^C]{0,5}', ig_chain2.sequence)
        # if rege1:
        #     chain2=rege1.group(1)
        #     rege2=re.match(r'.{0,7}[FW](G.*)', ig_chain3.sequence)
        #     if rege2:
        #         chain3=rege2.group(1)
        #         finalseq=chain2+finalseq+chain3
        #
        #
        #         final_ig =bcr.IgChain(finalseq, template_db=template_db, pdb_db=pdb_db)
        # #Score
        #         try:
        #             final_ig.hmmsearch(*hmms)
        #
        #         except Exception:
        #             pass
        #     try:
        #         aligned_seq=final_ig.aligned_seq
        #         print(aligned_seq)
        #     except:
        #         print("Error: Could not find aligned_seq")
        #         pass
