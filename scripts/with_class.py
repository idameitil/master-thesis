import numpy as np
import subprocess
import os
import sys
from Bio import SeqIO
from Bio import PDB
from joblib import Parallel, delayed
import multiprocessing as mp
import time

class energy_calculation:

    def __init__(self, model_filename, model_dir):
        self.model_filename = model_filename
        self.model_dir = model_dir
        self.model_ID = model_filename.replace(".pdb","")
        self.model_path = model_dir + model_filename
        self.output_dir = "/home/ida/master-thesis/testdir/"
        self.model_output_dir = self.output_dir + self.model_ID

    def pipeline(self):
        start_time = time.time()

        # Make output directory
        subprocess.run(["mkdir", self.model_output_dir])

        # Run FoldX
        try:
            run_foldx()
        except Exception as err:
            print("FoldX failed for: " + self.model_ID, file=sys.stderr)
            print(err, file=sys.stderr)

        # Extract foldX energies
        try:
            extract_foldx_energies()
        except Exception as err:
            print("Extracting foldX energies failed for: " + self.model_ID, file=sys.stderr)
            print(err, file=sys.stderr)

        # Run Rosetta
        try:
            rosetta_scorefile_path, rosetta_per_res_scorefile_path = run_rosetta(model_filename)
        except Exception as err:
            print("Rosetta failed for: " + self.model_ID, file=sys.stderr)
            print(err, file=sys.stderr)

        # Extract Rosetta energies
        try:
            rosetta_overall_scores, rosetta_per_res_scores = extract_rosetta_energies(rosetta_scorefile_path,
                                                                                      rosetta_per_res_scorefile_path,
                                                                                      model_filename)
        except Exception as err:
            print("Extracting Rosetta energies failed for: " + self.model_ID, file=sys.stderr)
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

    def run_foldx(self):

        # RepairPDB
        repair_command = "foldx --command=RepairPDB --pdb={} --ionStrength=0.05 --pH=7 --water=CRYSTAL --vdwDesign=2 --out-pdb=1 --pdbHydrogens=false --output-dir={}".format(self.model_filename, self.model_output_dir)
        subprocess.run(repair_command.split(), universal_newlines=True, stdout=subprocess.PIPE, cwd=self.model_dir)

        # AnalyseComplex
        repaired_pdb_path = self.model_output_dir + self.model_filename.replace(".fsa_model_TCR-pMHC.pdb", "_Repair.pdb")
        analyse_command = "foldx --command=AnalyseComplex --pdb={} --output-dir={}".format(repaired_pdb_path,
                                                                                           self.model_output_dir)
        subprocess.run(analyse_command.split(), universal_newlines=True, stdout=subprocess.PIPE, cwd=self.model_output_dir)

    def extract_foldx_energies(self):
        self.interaction_file_path = self.model_output_dir + "Interaction_" + model_filename.replace(".fsa_model_TCR-pMHC.pdb",
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
        self.foldx_interaction_energies = foldx_interaction_energies

    def run_rosetta(self, infile):

        # Relaxation
        rosetta_relax_command = "relax.default.linuxgccrelease \
                                -ignore_unrecognized_res \
                                -nstruct 1 \
                                -s {} \
                                -out:path:pdb {}".format(infile, self.model_output_dir)
        subprocess.run(rosetta_relax_command.split(), universal_newlines=True)

        # Scoring
        relaxed_pdb_path = self.model_output_dir + infile.replace(".pdb", "_0001.pdb")
        self.rosetta_scorefile_path = self.model_output_dir + model_filename + "_score.sc"
        rosetta_score_command = "score_jd2.linuxgccrelease \
                                -in:file:s {} \
                                -out:file:scorefile {}".format(relaxed_pdb_path, self.rosetta_scorefile_path)
        subprocess.run(rosetta_score_command.split(), universal_newlines=True)

        # Per residue scoring
        rosetta_per_res_scorefile_path = rosetta_output_dir + model_filename + "_per_residue_score.sc"
        rosetta_per_res_score_command = "per_residue_energies.linuxgccrelease \
                                        -in:file:s {} \
                                        -out:file:silent {}".format(relaxed_pdb_path, rosetta_per_res_scorefile_path)
        subprocess.run(rosetta_per_res_score_command.split(), universal_newlines=True)

        return rosetta_scorefile_path, rosetta_per_res_scorefile_path

    @staticmethod
    def oneHot(residue):
        mapping = dict(zip("ACDEFGHIKLMNPQRSTVWY", range(20)))
        if residue in "ACDEFGHIKLMNPQRSTVWY":
            return np.eye(20)[mapping[residue]]
        else:
            return np.zeros(20)

    @staticmethod
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

        return sequences, chains, total_length, numbering

    @staticmethod
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

    def splitPDB(self):

        model_path = model_dir + self.model_filename

        selectChain(ifn=model_path, ofn="TCR.pdb", chainID=["A", "B"])
        selectChain(ifn=model_path, ofn="pMHC.pdb", chainID=["M", "P"]))

e1 = energy_calculation("model_TCR-pMHC.pdb", "/home/ida/master-thesis/data/example_sequences/3TFK_model/")
e1.pipeline()