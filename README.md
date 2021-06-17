# master-thesis
Ida Meitil's master thesis - Using deep learning for improving TCR homology modeling and its application to immunogenicity prediction

The written report is found in _IdaMeitil_master_thesis.pdf_ 

## Data

* _data/NetTCR2_data_ : Original dataset from NetTCR2 paper

* _data/VDJdb_minscore0_2021-02-11.tsv_ : VDJdb download used in _scripts/mapping_tcrs.py_ to retrieve gene information for the positives

* _data/train_data_gene_names.csv_ : All entries from the sequence dataset. Containing the following information: ID, CDR3s, peptide, partition, binder/nonbinder, origin (10x, swapped or positives) and germlines.

* _data/TCR_genes.fasta_ : Fasta file of all V and J sequences, used in scripts/TCR_gene2seq.py for retrieving the full TCR sequences

* _data/train_data_full_sequences.csv_ : Contains the ID, TCRa sequence, TCRb sequence, peptide sequence, partition, binder/nonbinder, origin

* _data/HLA-02-01.fasta_ : Sequence of HLA-A*02-01-01 used for all the entries for the modeling

* _data/train_data_ : Full training dataset with sequence and energy terms. One .npy file for each entry. For example, the filename _4166_3p_neg_tenx.npy_ means that it is the entry with ID 4166, it is in partition 3, it's a negative and it's from the 10x dataset. The entries are not padded. The shape of each array is (x, 142), where x is the length of the sequence and 142 is the number of features. The features are described below.

Number        | Feature
------------- | -------------
1-20          | One-hot encoding of amino acid
21-24         | One-hot encoding of chain (which chain the residue is part of, MHC, peptide, TCRa or TCRb)
25-44         | Rosetta per-residue energy terms complex
45-64         | Rosetta per-residue energy terms separated complex
65-70         | FoldX interaction energy terms (constant for all residues)
71            | Rosetta total energy complex (constant for all residues)
72-94         | Rosetta individual break down energy terms complex (constant for all residues)
95            | Rosetta total energy TCR (constant for all residues)
96-118        | Rosetta individual break down energy terms TCR (constant for all residues)
119           | Rosetta total energy pMHC (constant for all residues)
120-142       | Rosetta individual break down energy terms pMHC (constant for all residues)

## Scripts

## Results

