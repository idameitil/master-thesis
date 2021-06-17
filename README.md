# master-thesis
Ida Meitil's master thesis - Using deep learning for improving TCR homology modeling and its application to immunogenicity prediction

* _IdaMeitil_master_thesis.pdf_: the written report

## Data

* _data/NetTCR2_data_: Original dataset from NetTCR2 paper

* _data/VDJdb_minscore0_2021-02-11.tsv_: VDJdb download used in _scripts/mapping_tcrs.py_ to retrieve gene information for the positives

* _data/train_data_gene_names.csv_: All entries from the sequence dataset. Containing the following information: ID, CDR3s, peptide, partition, binder/nonbinder, origin (10x, swapped or positives) and germlines.

* _data/TCR_genes.fasta_ : Fasta file of all V and J sequences, used in scripts/TCR_gene2seq.py for retrieving the full TCR sequences

* _data/train_data_full_sequences.csv_: Contains the ID, TCRa sequence, TCRb sequence, peptide sequence, partition, binder/nonbinder, origin

* _data/HLA-02-01.fasta_: Sequence of HLA-A*02-01-01 used for all the entries for the modeling

* _data/train_data_ : Full training dataset with sequence and energy terms. One .npy file for each entry. _4166_3p_neg_tenx.npy_ 

## Scripts

## Results

