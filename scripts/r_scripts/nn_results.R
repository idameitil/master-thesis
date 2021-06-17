
library(reshape2)

all_features_all_sequence <- read.csv("/home/ida/master-thesis/results/May25/all_features_all_sequence.csv")

all_features_no_mhc <- read.csv("/home/ida/master-thesis/results/May25/all_features_no_MHC.csv")
all_features_no_mhc$X.Comment <- paste(all_features_no_mhc$X.Comment, " (", all_features_no_mhc$count, ")", sep = "")

only_AAseq_no_mhc <- read.csv("/home/ida/master-thesis/results/May25/only_AAseq_no_MHC.csv")
only_AAseq_no_mhc$X.Comment <- paste(only_AAseq_no_mhc$X.Comment, " (", only_AAseq_no_mhc$count, ")", sep = "")

AAseq_totalRos_no_mhc <- read.csv("/home/ida/master-thesis/results/May25/onehot_totalenergyterms_no_MHC.csv")
AAseq_totalRos_no_mhc$X.Comment <- paste(AAseq_totalRos_no_mhc$X.Comment, " (", AAseq_totalRos_no_mhc$count, ")", sep = "")

global_no_mhc <- read.csv("/home/ida/master-thesis/results/May25/globalenergyterms_no_MHC.csv")
global_no_mhc$X.Comment <- paste(global_no_mhc$X.Comment, " (", global_no_mhc$count, ")", sep = "")

energy_terms_no_mhc <- read.csv("/home/ida/master-thesis/results/May25/only_energyterms_no_MHC.csv")
energy_terms_no_mhc$X.Comment <- paste(energy_terms_no_mhc$X.Comment, " (", energy_terms_no_mhc$count, ")", sep = "")

df1 <- cbind(all_features_no_mhc[,c("X.Comment", "AUC")], rep("all", 19))
colnames(df1) <- c("peptide", "AUC", "feature_set")
df2 <- cbind(only_AAseq_no_mhc[,c("X.Comment", "AUC")], rep("seq", 19))
colnames(df2) <- c("peptide", "AUC", "feature_set")
df3 <- cbind(AAseq_totalRos_no_mhc[,c("X.Comment", "AUC")], rep("seq+totalRosetta", 19))
colnames(df3) <- c("peptide", "AUC", "feature_set")
df4 <- cbind(global_no_mhc[,c("X.Comment", "AUC")], rep("global_energies", 19))
colnames(df4) <- c("peptide", "AUC", "feature_set")
df5 <- cbind(energy_terms_no_mhc[,c("X.Comment", "AUC")], rep("all_energies", 19))
colnames(df5) <- c("peptide", "AUC", "feature_set")

new_df <- rbind(df1, df2, df3, df4, df5)
new_df$feature_set <- as.factor(new_df$feature_set)
new_df$peptide <- as.factor(new_df$peptide)

new_df_melt <- melt(new_df)

ordered_df <- all_features_no_mhc[
  with(all_features_no_mhc, order(-count)),
]
peptide_order <- ordered_df$X.Comment

feature_set_colors <- brewer.pal(5, "Paired")

new_df_melt$feature_set <- factor(new_df_melt$feature_set, levels = c("all", "seq", "all_energies", "seq+totalRosetta", "global_energies"))
levels(new_df_melt$feature_set) <- c("All features", "Only sequence", "Only energy terms", "Sequence and \ntotal Rosetta terms", "Only global\n energy terms")
pdf("/home/ida/master-thesis/results/parts_of_features.pdf", height = 4)
ggplot(new_df_melt[new_df_melt$peptide != "Total (33048)" &
                     new_df_melt$peptide != "CLGGLLTMV (28)" &
                     new_df_melt$peptide != "ILKEPVHGV (16)",], 
       aes(x = factor(peptide, level = peptide_order), y = value, fill = feature_set)) +
  geom_bar(position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Peptide") +
  ylab("AUC") +
  scale_fill_manual(values = feature_set_colors) +
  geom_text(aes(label=round(value, 2)), vjust=-0.3, 
            position = position_dodge(0.9))
dev.off()

# Only total
pdf("/home/ida/master-thesis/results/parts_of_features_total.pdf", height = 3, width = 3)
ggplot(new_df_melt[new_df_melt$peptide == "Total (33048)",], 
       aes(x = feature_set, y = value)) +
  geom_bar(position = "dodge", stat = "identity", fill = "grey") +
  geom_text(aes(label=round(value, 2)), vjust=-0.25) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  xlab("Feature set") +
  ylab("AUC") +
  ylim(0,0.9) +
  scale_x_discrete(limits = c("all", "seq", "all_energies", "seq+totalRosetta", "global_energies"),
                   labels = c("All features", "Only sequence", "Only energy terms", "Sequence and \ntotal Rosetta terms", "Only global\n energy terms"))
dev.off()

# Only total MCC
df1 <- cbind(all_features_no_mhc[,c("X.Comment", "MCC")], rep("all", 19))
colnames(df1) <- c("peptide", "MCC", "feature_set")
df2 <- cbind(only_AAseq_no_mhc[,c("X.Comment", "MCC")], rep("seq", 19))
colnames(df2) <- c("peptide", "MCC", "feature_set")
df3 <- cbind(AAseq_totalRos_no_mhc[,c("X.Comment", "MCC")], rep("seq+totalRosetta", 19))
colnames(df3) <- c("peptide", "MCC", "feature_set")
df4 <- cbind(global_no_mhc[,c("X.Comment", "MCC")], rep("global_energies", 19))
colnames(df4) <- c("peptide", "MCC", "feature_set")
df5 <- cbind(energy_terms_no_mhc[,c("X.Comment", "MCC")], rep("all_energies", 19))
colnames(df5) <- c("peptide", "MCC", "feature_set")

new_df_MCC <- rbind(df1, df2, df3, df4, df5)
new_df$feature_set <- as.factor(new_df$feature_set)
new_df$peptide <- as.factor(new_df$peptide)

new_df_MCC_melt <- melt(new_df_MCC)
pdf("/home/ida/master-thesis/results/parts_of_features_total_MCC.pdf", height = 3, width = 3)
ggplot(new_df_MCC_melt[new_df_MCC_melt$peptide == "Total (33048)",], 
       aes(x = feature_set, y = value)) +
  geom_bar(position = "dodge", stat = "identity", fill = "grey") +
  geom_text(aes(label=round(value, 2)), vjust=-0.25) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  xlab("Feature set") +
  ylab("MCC") +
  ylim(0,0.75) +
  scale_x_discrete(limits = c("all", "seq", "all_energies", "seq+totalRosetta", "global_energies"),
                   labels = c("All features", "Only sequence", "Only energy terms", "Sequence and \ntotal Rosetta terms", "Only global\n energy terms"))
dev.off()

# Evaluation on swapped vs 10x
all_features <- c(0.82, 0.82, 0.83)
seq <- c(0.86, 0.89, 0.85)
evaluation_set <- c("All", "Positive + swapped negatives", "Positives + 10x negatives")
eval_swap_10x <- data.frame(all_features, seq, evaluation_set)
eval_melt <- melt(eval_swap_10x)
colnames(eval_melt) <- c("Evaluation_set", "Features", "AUC")
pdf("/home/ida/master-thesis/results/swapped_vs_tenx_eval.pdf", height = 3, width = 5)
ggplot(eval_melt, aes(x = Features, y = AUC, fill = Evaluation_set)) +
  geom_bar(position = "dodge", stat = "identity") +
  scale_fill_discrete(name = "Evaluation complexes") +
  scale_x_discrete(labels = c("All features", "Only sequence"))
dev.off()
