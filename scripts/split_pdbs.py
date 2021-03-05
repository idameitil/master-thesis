from Bio import PDB

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

selectChain(ifn='/home/ida/master-thesis/data/example_sequences/3TFK_model/model_TCR-pMHC.pdb', ofn="TCR.pdb", chainID=["A","B"])
selectChain(ifn='/home/ida/master-thesis/data/example_sequences/3TFK_model/model_TCR-pMHC.pdb', ofn="pMHC.pdb", chainID=["M","P"])