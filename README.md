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

* _data/example_array.csv_ : Example of an entry from the dataset converted to a csv-file for inspecting the data.

## Scripts

* _mapping_tcrs.py_ : Takes input file with CDR3 sequences and maps them onto VDJdb download (_data/VDJdb_minscore0_2021-02-11.tsv_) for retrieval of V and J genes. Writes output file with gene names.
* _swap_peptides.py_ : Generates swapped negatives. For each positive, the TCR is combined with a different peptide from the dataset based on the frequencies of the peptides. This is done in a very complicated way, and it might be better to just pair it with a random peptide.
* _TCR_gene2seq.py_ : Retrieves the full TCR sequences. Takes input file with gene names. Retrieves gene sequences from _data/TCR_genes.fasta_. Combines gene sequences with CDR3 and aligns to precompiled HMM from Lyra. Writes output file with ID and full TCRa and TCRb sequences. Requires Lyra (bcr-models).
* _modeling.py_ : Performs molecular modeling. Input file is a csv file with all entries with full TCRa and TCRb sequences and peptide sequence. Writes fasta file for each entry. Runs TCRpMHCmodels for each fasta. Run in parallel. Requires TCRpMHCmodels which is installed on computerome in the environment _/home/projects/ht3_aim/people/idamei/scripts/tcrpmhc_env2_ . The output is a folder for each entry located on computerome at _/home/projects/ht3_aim/people/idamei/results/modeling_output/_ . From each output folder only the file _model_TCR-pMHC.pdb_ is used.
* _energy_calc_pipeline.py_ : Performs energy calculations and creates final dataset. For each model the following things are down: 
  * A PDB is created for TCR and pMHC separeted
  * The structure is relaxed and scored in FoldX
  * The interaction energies are extracted from the FoldX output
  * The structure is relaxed and scored in Rosetta (global and per-residues). The TCR and pMHC separeted are relaxed and scored in Rosetta.
  * The Rosetta energy terms are extracted
  * A .npy file is created containing: One-hot encoding of sequence, one-hot encoding of chain, and energy terms
  Run in parallel. Took about 4 days to run, when using 4 cores on computerome. Sometimes fails when running many at a time, therefor I ran it in smaller parts.
* _pad_select_features_combine_in_one_array.py_ :  For each entry, the array is padded so they all have same length and a number of features are selected (see report). For each partition, an input file, a label file, an origin file, and a peptide file (for per-peptide performance) is created. Could be changed to pad TCRa and TCRb separately so that they can later be retrieved.
* _CNN_LSTM.py_ : CNN-biLSTM network. Takes the padded dataset as input (_/home/projects/ht3_aim/people/idamei/data/train_data2/_). Different subsets of features and residues are specified in the "Load data" section. There are three modes:
   * "simple_train": the model is trained on the first four partitions and with early-stopping on the fifth partition
   * "nested_cross_val": Model is trained in a nested cross validation setup. Both total performance and per-peptide performance is written to the output csvfile
   * "leave_one_out": For each peptide, the model is trained in a nested cross val setup, without the given peptide in the training set, and performance is calculated for the peptide.
* siamese.py: my adaption of Magnus' siamese network to a more simple training setup.


## Results

