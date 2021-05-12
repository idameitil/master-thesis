
__requires__ = 'bcr-models==0.0.0'
import bcr_models as bcr
import re
import csv
from Bio import SeqIO

##################################################
### Make dictionary of TCR genes and sequences ###
##################################################

gene_sequences = SeqIO.parse(open('TCR_genes.fasta'),'fasta')
genes = {}
for fasta in gene_sequences:
    name, sequence = fasta.id, str(fasta.seq)
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
        genes[(acti, actt, actf, actg, acta)] = sequence
    else:
        print("Error. Could not understand gene name in database file:" + name)

###############################################
### Retrieve full sequences for input file ###
###############################################

hmms = bcr.db.builtin_hmms()
template_db = bcr.db.BuiltinTemplateDatabase()
pdb_db = bcr.db.BuiltinPDBDatabase()

count_success = 0
error_1, error_2, error_3 = 0, 0, 0

input_filename = sys.argv[1]
#ID,CDR3a,CDR3b,v_gene_alpha,j_gene_alpha,v_gene_beta,j_gene_beta
output_filename = sys.argv[2]
outfile = open(output_filename, "w")
outfile.write("#ID,TCRa_sequence,TCRb_sequence,peptide,partition,binder\n")
print(inputfilename)
print(output_filename)
with open(input_filename, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[0].startswith("#"):
            continue
        ID, CDR3a, CDR3b, peptide, partition, binder, v_gene_alpha, j_gene_alpha, v_gene_beta, j_gene_beta= rows

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
                vseq=genes[(acti_v,actt_v,actf_v,actg_v,acta_v)]
            except:
                print("Error. Could not find V gene in database: " + v_gene_alpha, str(acti_v),str(actt_v),str(actf_v),str(actg_v),str(acta_v))
                continue
        else:
            print("Error: Could not recognize V gene: {}".format(v_gene_alpha))
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
                jseq=genes[(acti_j,actt_j,actf_j,actg_j,acta_j)]
            except:
                print("Error. Could not find J gene in database: " + j_gene_alpha + acti_j,actt_j,actf_j,actg_j,acta_j)
                continue
        else:
            print("Error: Could not recognize J gene: {}".format(j_gene_alpha))
            continue

        ig_chain1 =bcr.IgChain(CDR3a, template_db=template_db, pdb_db=pdb_db)
        ig_chain2 =bcr.IgChain(vseq, template_db=template_db, pdb_db=pdb_db)
        ig_chain3 =bcr.IgChain(jseq, template_db=template_db, pdb_db=pdb_db)

        rege1=re.match(r'(.+C)[^C]{0,5}', ig_chain2.sequence)
        if rege1:
            chain2=rege1.group(1)
            rege2=re.match(r'.{0,11}([FW]G.*)', ig_chain3.sequence)
            if rege2:
                chain3=rege2.group(1)
                finalseq=chain2+CDR3a+chain3
                final_ig =bcr.IgChain(finalseq, template_db=template_db, pdb_db=pdb_db)
                try:
                    final_ig.hmmsearch(*hmms)
                    aligned_seq=final_ig.aligned_seq
                    aligned_seq_alpha = aligned_seq
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
                vseq=genes[(acti_v,actt_v,actf_v,actg_v,acta_v)]
            except:
                print("Error. Could not find V gene in database: " + v_gene_beta, str(acti_v),str(actt_v),str(actf_v),str(actg_v),str(acta_v))
                continue
        else:
            print("Error: Could not recognize V gene: {}".format(v_gene_beta))
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
                jseq=genes[(acti_j,actt_j,actf_j,actg_j,acta_j)]
            except:
                print("Error. Could not find J gene in database: " + j_gene_beta + acti_j,actt_j,actf_j,actg_j,acta_j)
                continue
        else:
            print("Error: Could not recognize J gene: {}".format(j_gene_beta))
            continue

        ig_chain1 =bcr.IgChain(CDR3b, template_db=template_db, pdb_db=pdb_db)
        ig_chain2 =bcr.IgChain(vseq, template_db=template_db, pdb_db=pdb_db)
        ig_chain3 =bcr.IgChain(jseq, template_db=template_db, pdb_db=pdb_db)

        rege1=re.match(r'(.+C)[^C]{0,5}', ig_chain2.sequence)
        if rege1:
            chain2=rege1.group(1)
            rege2=re.match(r'.{0,11}([FW]G.*)', ig_chain3.sequence)
            if rege2:
                chain3=rege2.group(1)
                finalseq=chain2+CDR3b+chain3
                final_ig =bcr.IgChain(finalseq, template_db=template_db, pdb_db=pdb_db)
                try:
                    final_ig.hmmsearch(*hmms)
                    aligned_seq=final_ig.aligned_seq
                    aligned_seq_beta = aligned_seq
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
        # ID,TCRa_sequence,TCRb_sequence,peptide,partition,binder
        if aligned_seq_alpha == '' or aligned_seq_beta == '':
            pass
        else:
            TCRa_seq = aligned_seq_alpha.replace("-", "")
            TCRb_seq = aligned_seq_beta.replace("-", "")
            outfile.write('{},{},{},{},{},{}\n'.format(ID,TCRa_seq,TCRb_seq,peptide,partition,binder))
print("Success: " + str(count_success))
print("Error hmmsearch failed: " + str(error_1))
print("Error F/W in J gene: " + str(error_2))
print("Error C in V gene: " + str(error_3))
