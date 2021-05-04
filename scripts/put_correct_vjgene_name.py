import re
import sys
import os
import pandas

from pprint import pprint
import re
import csv

from Bio import SeqIO
import argparse

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
        genes[(acti,actt,actf,actg,acta)]=(sequence, name)
    else:
        print("Error. Could not understand name in database file:" + name)

filein = "/home/ida/master-thesis/data/temporary_data/all_data_numbered_origin.csv"
outfile = open("/home/ida/master-thesis/data/temporary_data/all_data_numbered_origin_vdjdbnames.csv", "w")

outfile.write("#ID,CDR3a,CDR3b,peptide,partition,binder,v_gene_alpha,j_gene_alpha,v_gene_beta,j_gene_beta,origin,v_alpha_vdjdb_name, j_alpha_vdjdb_name, v_beta_vdjdb_name, j_beta_vdjdb_name\n")

with open(filein, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[0].startswith("#"):
            continue
        ID, CDR3a, CDR3b, peptide, partition, binder, v_gene_alpha, j_gene_alpha, v_gene_beta, j_gene_beta, origin = rows

        ### Find alpha sequence ###
        vseq=''
        jseq=''
        aligned_seq_alpha = ''
        # Find V sequence alpha
        rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', v_gene_alpha)
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
                (vseq, vname_alpha) = genes[(acti_v,actt_v,actf_v,actg_v,acta_v)]
            except:
                print("Error. Could not find V gene in database: " + v_gene_alpha, str(acti_v),str(actt_v),str(actf_v),str(actg_v),str(acta_v))
                vname_alpha = v_gene_alpha + "_not_found"
                continue
        else:
            print("Error: Could not recognize V gene: {}".format(v_gene_alpha))
            vname_alpha = v_gene_alpha + "_not_found"
            continue

        # Find J sequence alpha
        rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', j_gene_alpha)
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
                (jseq, jname_alpha) = genes[(acti_j,actt_j,actf_j,actg_j,acta_j)]
            except:
                print("Error. Could not find J gene in database: " + j_gene_alpha + acti_j,actt_j,actf_j,actg_j,acta_j)
                jname_alpha = j_gene_alpha + "_not_found"
                continue
        else:
            print("Error: Could not recognize J gene: {}".format(j_gene_alpha))
            jname_alpha = j_gene_alpha + "_not_found"
            continue

        ### Find beta sequence ###
        vseq=''
        jseq=''
        aligned_seq_beta = ''
        # Find V sequence beta
        rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', v_gene_beta)
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
                (vseq, vname_beta) = genes[(acti_v,actt_v,actf_v,actg_v,acta_v)]
            except:
                print("Error. Could not find V gene in database: " + v_gene_beta, str(acti_v),str(actt_v),str(actf_v),str(actg_v),str(acta_v))
                vname_beta = v_gene_beta + "_not_found"
                continue
        else:
            print("Error: Could not recognize V gene: {}".format(v_gene_beta))
            vname_beta = v_gene_beta + "_not_found"
            continue

        # Find J sequence beta
        rege = re.match(r'TR(.)(.)0*(\d+)([^\|]+)?', j_gene_beta)
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
                (jseq, jname_beta) = genes[(acti_j,actt_j,actf_j,actg_j,acta_j)]
            except:
                print("Error. Could not find J gene in database: " + j_gene_beta + acti_j,actt_j,actf_j,actg_j,acta_j)
                jname_beta = j_gene_beta + "_not_found"
                continue
        else:
            print("Error: Could not recognize J gene: {}".format(j_gene_beta))
            jname_beta = j_gene_beta + "_not_found"
            continue
        outfile.write(f"{ID},{CDR3a},{CDR3b},{peptide},{partition},{binder},{v_gene_alpha},{j_gene_alpha},{v_gene_beta},{j_gene_beta},{origin},{vname_alpha},{jname_alpha},{vname_beta},{jname_beta}\n")




