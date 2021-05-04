library(stringr)
library(dplyr)
library(reshape2)
library(ggplot2)

### LOAD DATA ###
data <- read.csv("/home/ida/master-thesis/data/temporary_data/all_data_numbered_origin_vdjdbnames.csv")
data$cdr3a_loop_length <- as.numeric(lapply(data$CDR3a, str_length))
data$cdr3b_loop_length <- as.numeric(lapply(data$CDR3b, str_length))

### COUNTS ###
unique(data$peptide)
length(unique(data$peptide))
length(unique(data$CDR3a))
length(unique(data$CDR3b))

### BARPLOTS OF GERM LINES ###
pdf("/home/ida/master-thesis/results/0503_sequence_statistics/j_gene_alpha.pdf", width = 10)
ggplot(data = data, aes(x = j_alpha_vdjdb_name, fill = origin)) +
  geom_bar(position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
dev.off()

pdf("/home/ida/master-thesis/results/0503_sequence_statistics/v_gene_alpha.pdf", width = 10)
ggplot(data = data, aes(x = v_alpha_vdjdb_name, fill = origin)) +
  geom_bar(position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
dev.off()

pdf("/home/ida/master-thesis/results/0503_sequence_statistics/j_gene_beta.pdf", width = 10)
ggplot(data = data, aes(x = j_beta_vdjdb_name, fill = origin)) +
  geom_bar(position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
dev.off()

pdf("/home/ida/master-thesis/results/0503_sequence_statistics/v_gene_beta.pdf", width = 10)
ggplot(data = data, aes(x = v_beta_vdjdb_name, fill = origin)) +
  geom_bar(position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
dev.off()

### HISTOGRAMS OF LOOP LENGTH ###
pdf("/home/ida/master-thesis/results/0503_sequence_statistics/loop_length_cdr3a.pdf", height = 4)
ggplot(data, aes(x = cdr3a_loop_length)) +
  geom_histogram(binwidth = 0.5) +
  facet_grid(cols = vars(origin))
dev.off()

pdf("/home/ida/master-thesis/results/0503_sequence_statistics/loop_length_cdr3b", height = 4)
ggplot(data, aes(x = cdr3b_loop_length)) +
  geom_histogram(binwidth = 0.5) +
  facet_grid(cols = vars(origin))
dev.off()

