#!/usr/bin/env python
# coding: utf-8

import numpy as np
import subprocess
import os
import sys
from Bio import SeqIO
from Bio import PDB
from joblib import Parallel, delayed
import multiprocessing as mp
import time

def oneHot(residue):
    mapping = dict(zip("ACDEFGHIKLMNPQRSTVWY", range(20)))
    if residue in "ACDEFGHIKLMNPQRSTVWY":
        return np.eye(20)[mapping[residue]]
    else:
        return np.zeros(20)


def reverseOneHot(encoding):
    mapping = dict(zip(range(20), "ACDEFGHIKLMNPQRSTVWY"))
    seq = ''
    for i in range(len(encoding)):
        if np.max(encoding[i]) > 0:
            seq += mapping[np.argmax(encoding[i])]
    return seq


def extract_seq_pdb(pdb_path):
    pdb_sequence = SeqIO.parse(open(pdb_path), 'pdb-atom')
    sequences = []
    chains = []
    total_length = 0
    for chain in pdb_sequence:
        sequences.append(str(chain.seq))
        total_length += len(str(chain.seq))
        chains.append(chain.id.replace('????:', ''))

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

    print(sequences)

    return sequences, chains, total_length, numbering

def selectChain(ifn, ofn, chainID):
    """Saves selected chains from PDB in a new PDB"""
    parser = PDB.PDBParser()
    structure = parser.get_structure('x', ifn)

    class ChainSelector():
        def __init__(self, chainID=chainID):
            self.chainID = chainID

        def accept_chain(self, chain):
            if chain.get_id() in self.chainID:
                return 1
            return 0
        def accept_model(self, model):
            return 1
        def accept_residue(self, residue):
            return 1
        def accept_atom(self, atom):
            return 1

    sel = ChainSelector(chainID)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(ofn, sel)

def splitPDB(model_filename):

    model_path = model_dir + model_filename

    selectChain(ifn=model_path, ofn="TCR.pdb", chainID=["A","B"])
    selectChain(ifn=model_path, ofn="pMHC.pdb", chainID=["M","P"]))

def run_foldx(model_filename):
    # Make output directory
    foldx_output_dir = "/home/people/idamei/energy_calc_pipeline/foldx_output/" \
                       + model_filename + "_foldx/"
    subprocess.run(["mkdir", foldx_output_dir])

    # RepairPDB
    # repair_command = "foldx --command=RepairPDB --pdb={} --ionStrength=0.05 --pH=7 --water=CRYSTAL --vdwDesign=2 --out-pdb=1 --pdbHydrogens=false --output-dir={}".format(model_filename, foldx_output_dir)
    # subprocess.run(repair_command.split(), universal_newlines=True, stdout=subprocess.PIPE, cwd=model_dir)

    # AnalyseComplex
    repaired_pdb_path = model_filename.replace(".fsa_model_TCR-pMHC.pdb", "_Repair.pdb")
    analyse_command = "foldx --command=AnalyseComplex --pdb={} --output-dir={}".format(repaired_pdb_path,
                                                                                       foldx_output_dir)
    subprocess.run(analyse_command.split(), universal_newlines=True, stdout=subprocess.PIPE, cwd=foldx_output_dir)

    return foldx_output_dir


def extract_foldx_energies(foldx_output_dir, model_filename):
    interaction_file_path = foldx_output_dir + "Interaction_" + model_filename.replace(".fsa_model_TCR-pMHC.pdb",
                                                                                       "_Repair_AC.fxout")
    foldx_output = open(interaction_file_path, "r")
    foldx_interaction_energies = dict()
    for line in foldx_output:
        if line.startswith("./"):
            splitted_line = line.split("\t")
            group1 = splitted_line[1]
            group2 = splitted_line[2]
            interaction_energy = splitted_line[6]
            foldx_interaction_energies[group1 + group2] = float(interaction_energy)
    foldx_output.close()
    return foldx_interaction_energies

def run_rosetta(model_filename):
    model_path = model_dir + model_filename
    rosetta_output_dir = "/home/people/idamei/energy_calc_pipeline/rosetta_output/" \
                         + model_filename + "_rosetta/"
    subprocess.run(["mkdir", rosetta_output_dir])

    # Relaxation
    rosetta_relax_command = "relax.default.linuxgccrelease \
                            -ignore_unrecognized_res \
                            -nstruct 1 \
                            -s {} \
                            -out:path:pdb {}".format(model_path, rosetta_output_dir)

    # subprocess.run(rosetta_relax_command.split(), universal_newlines=True)

    # Scoring
    relaxed_pdb_path = rosetta_output_dir + model_filename.replace(".pdb", "_0001.pdb")
    rosetta_scorefile_path = rosetta_output_dir + model_filename + "_score.sc"
    rosetta_score_command = "score_jd2.linuxgccrelease \
                            -in:file:s {} \
                            -out:file:scorefile {}".format(relaxed_pdb_path, rosetta_scorefile_path)
    # subprocess.run(rosetta_score_command.split(), universal_newlines=True)

    # Per residue scoring
    rosetta_per_res_scorefile_path = rosetta_output_dir + model_filename + "_per_residue_score.sc"
    rosetta_per_res_score_command = "per_residue_energies.linuxgccrelease \
                                    -in:file:s {} \
                                    -out:file:silent {}".format(relaxed_pdb_path, rosetta_per_res_scorefile_path)
    # subprocess.run(rosetta_per_res_score_command.split(), universal_newlines=True)

    return rosetta_scorefile_path, rosetta_per_res_scorefile_path

def extract_rosetta_energies(rosetta_scorefile_path, rosetta_per_res_scorefile_path, model_filename):
    # Rosetta overall energies
    rosetta_scorefile = open(rosetta_scorefile_path, "r")
    rosetta_scorefile.readline()
    rosetta_scorefile.readline()
    # SCORE: total_score       score dslf_fa13    fa_atr    fa_dun   fa_elec fa_intra_rep fa_intra_sol_xover4              fa_rep              fa_sol hbond_bb_sc hbond_lr_bb    hbond_sc hbond_sr_bb linear_chainbreak lk_ball_wtd       omega overlap_chainbreak            p_aa_pp pro_close rama_prepro         ref        time yhh_planarity description
    line = rosetta_scorefile.readline()
    splitted_line = line.strip().split()
    rosetta_overall_scores = splitted_line[1:-1]  # 24 elements
    rosetta_overall_scores = [float(x) for x in rosetta_overall_scores]
    rosetta_scorefile.close()

    # Rosetta per residue energies
    rosetta_per_res_scorefile = open(rosetta_per_res_scorefile_path, "r")
    # Find length of chains
    model_path = model_dir + model_filename
    sequences, chains, total_length, numbering = extract_seq_pdb(model_path)
    length_M = max(numbering["M"])
    length_P = max(numbering["P"])
    length_A = max(numbering["A"])
    length_B = max(numbering["B"])
    rosetta_per_res_scores = {"M": {}, "P": {}, "A": {}, "B": {}}
    # SCORE:     pose_id     pdb_id     fa_atr     fa_rep     fa_sol    fa_intra_rep    fa_intra_sol_xover4    lk_ball_wtd    fa_elec    pro_close    hbond_sr_bb    hbond_lr_bb    hbond_bb_sc    hbond_sc    dslf_fa13      omega     fa_dun    p_aa_pp    yhh_planarity        ref    rama_prepro      score description
    for line in rosetta_per_res_scorefile:
        splitted_line = line.strip().split()
        if splitted_line[1] == "pose_id":
            continue
        pdb_id = splitted_line[2]
        chain = pdb_id[-1]
        position = int(pdb_id[:-1])
        rosetta_per_res_scores[chain][position] = [float(x) for x in splitted_line[3:-1]]  # 20 elements
    rosetta_scorefile.close()

    return rosetta_overall_scores, rosetta_per_res_scores


def create_output(model_filename, foldx_interaction_energies, rosetta_overall_scores, rosetta_per_res_scores):
    # one-hot AA, M, P, TCR, foldx_MP, foldx_MA, foldx_MB, foldx_PA, foldx_PB, foldx_AB,
    # Rosetta_total_energy, Rosetta_per_res_indiv_energies

    # Extract sequence from PDB
    model_path = model_dir + model_filename
    sequences, chains, total_length, numbering = extract_seq_pdb(model_path)

    # Create output_array
    output_array = np.empty(shape=(total_length, 74))
    k1 = 0  # chain
    k2 = 0  # residue number total
    for chain in sequences:
        chain = chains[k1]
        sequence = sequences[k1]
        k1 += 1
        k3 = 0  # chain residue number
        for aminoacid in sequence:
            number = numbering[chain][k1]
            output_array[k2, 0:20] = oneHot(aminoacid)
            if chain == "M":
                output_array[k2, 20:24] = np.array([1, 0, 0, 0])
                output_array[k2, 54:74] = rosetta_per_res_scores["M"][number]
            if chain == "P":
                output_array[k2, 20:24] = np.array([0, 1, 0, 0])
                output_array[k2, 54:74] = rosetta_per_res_scores["P"][number]
            if chain == "A":
                output_array[k2, 20:24] = np.array([0, 0, 1, 0])
                output_array[k2, 54:74] = rosetta_per_res_scores["A"][number]
            if chain == "B":
                output_array[k2, 20:24] = np.array([0, 0, 0, 1])
                output_array[k2, 54:74] = rosetta_per_res_scores["B"][number]
            output_array[k2, 24:30] = list(foldx_interaction_energies.values())
            output_array[k2, 30:54] = rosetta_overall_scores
            k2 += 1
            k3 += 1
    # np.set_printoptions(threshold=sys.maxsize)
    #   print(output_array)

    return output_array


def pipeline(model_filename):
    start_time = time.time()

    model_ID = model_filename.replace(".pdb", "")
    output_dir = "/home/people/idamei/results/model_energies/" + model_ID
    subprocess.run(["mkdir", output_dir])

    # Run FoldX
    try:
        foldx_output_dir = run_foldx(mo)
    except Exception as err:
        print("FoldX failed for: " + model_filename, file=sys.stderr)
        print(err, file=sys.stderr)

    # Extract foldX energies
    try:
        foldx_interaction_energies = extract_foldx_energies(foldx_output_dir, model_filename)
    except Exception as err:
        print("Extracting foldX energies failed for: " + model_filename, file=sys.stderr)
        print(err, file=sys.stderr)

    # Run Rosetta
    try:
        rosetta_scorefile_path, rosetta_per_res_scorefile_path = run_rosetta(model_filename)
    except Exception as err:
        print("Rosetta failed for: " + model_filename, file=sys.stderr)
        print(err, file=sys.stderr)

    # Extract Rosetta energies
    try:
        rosetta_overall_scores, rosetta_per_res_scores = extract_rosetta_energies(rosetta_scorefile_path,
                                                                                  rosetta_per_res_scorefile_path,
                                                                                  model_filename)
    except Exception as err:
        print("Extracting Rosetta energies failed for: " + model_filename, file=sys.stderr)
        print(err, file=sys.stderr)

    # Split pdb

    # Run Rosetta TCR
    # Exract Rosetta TCR

    # Run Rosetta TCR
    # Exract Rosetta TCR

    # Create output
    try:
        output_array = create_output(model_filename, foldx_interaction_energies, rosetta_overall_scores,
                                     rosetta_per_res_scores)
    except Exception as err:
        print("Creating output failes for: " + model_filename, file=sys.stderr)
        print(err, file=sys.stderr)

    runtime = (time.time() - start_time) / 60
    print("{} took {} min".format(model_filename, runtime))

    return output_array


model_dir = "/home/people/idamei/modeling/models/"
p = subprocess.Popen(["ls", model_dir],
                     stdout=subprocess.PIPE, universal_newlines=True)
models = p.communicate()[0].split()
if "molecules" in models:
    models.remove("molecules")
if "rotabase.txt" in models:
    models.remove("rotabase.txt")
models = ["3TFK.fsa_model_TCR-pMHC.pdb"]
pool = mp.Pool(4)

results = []
results.append(pool.map(pipeline, [model for model in models]))

pool.close()

# np.savez("/home/people/idamei/energy_calc_pipeline/energy_output_arrays.npz", results

