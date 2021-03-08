from Bio import SeqIO
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

def extract_pdb_features(pdb_path):

    # Get chain names
    pdb_sequence = SeqIO.parse(open(pdb_path), 'pdb-atom')
    chain_names = []
    for chain in pdb_sequence:
        #print(chain)
        chain_names.append(chain.id.replace('????:', ''))

    # Get sequences
    structure = PDBParser().get_structure('3TKF', pdb_path)
    ppb = PPBuilder()
    chain_sequences = {}
    i = 0
    for pp in ppb.build_peptides(structure):
        print(pp)
        chain_name = chain_names[i]
        chain_sequences[chain_name] = str(pp.get_sequence())
        i += 1

    length_A = len(chain_sequences["A"])
    length_B = len(chain_sequences["B"])
    length_M = len(chain_sequences["M"])
    length_P = len(chain_sequences["P"])
    total_length = length_P + length_M + length_B + length_A

    # Get sequence numbering
    pdb_file = open(pdb_path, "r")
    old_number = 0
    numbering = {"M": [], "P": [], "A": [], "B": []}
    for line in pdb_file:
        splitted_line = line.split()
        if splitted_line[0] != "ATOM":
            continue
        chain = splitted_line[4]
        new_number = splitted_line[5]
        if new_number != old_number:
            numbering[chain].append(int(new_number))
            old_number = new_number

    print(chain_names)
    print(chain_sequences)
    print(total_length)
    print(numbering)
    print(length_A, length_M, length_B, length_P)


extract_pdb_features("/home/ida/master-thesis/data/example_sequences/3TFK_model/model_TCR-pMHC.pdb")
#print(sequences, chains, total_length, numbering)
