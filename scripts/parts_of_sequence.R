library(reshape2)
library(ggplot2)
library(RColorBrewer)

residue_colors <- brewer.pal(4, "Spectral")

all_seq <- read.csv("/home/ida/master-thesis/results/May25/all_features_all_sequence.csv")
no_mhc <- read.csv("/home/ida/master-thesis/results/May25/all_features_no_MHC.csv")
no_peptide <- read.csv("/home/ida/master-thesis/results/May25/all_features_no_peptide.csv")
no_tcr <- read.csv("/home/ida/master-thesis/results/May25/all_features_no_TCR.csv")

all_seq$X.Comment <- paste(all_seq$X.Comment, " (", all_seq$count, ")", sep = "")
no_mhc$X.Comment <- all_seq$X.Comment
no_peptide$X.Comment <- all_seq$X.Comment
no_tcr$X.Comment <- all_seq$X.Comment

df1 <- cbind(all_seq[,c("X.Comment", "AUC")], rep("all", 19))
colnames(df1) <- c("peptide", "AUC", "residues")
df2 <- cbind(no_mhc[,c("X.Comment", "AUC")], rep("no_mhc", 19))
colnames(df2) <- c("peptide", "AUC", "residues")
df3 <- cbind(no_peptide[,c("X.Comment", "AUC")], rep("no_peptide", 19))
colnames(df3) <- c("peptide", "AUC", "residues")
df4 <- cbind(no_tcr[,c("X.Comment", "AUC")], rep("no_tcr", 19))
colnames(df4) <- c("peptide", "AUC", "residues")
new_df <- rbind(df1, df2, df3, df4)

new_df$feature_set <- as.factor(new_df$residues)
new_df$peptide <- as.factor(new_df$peptide)
new_df_melt <- melt(new_df)

ordered_df <- all_seq[
  with(all_seq, order(-count)),
]
peptide_order <- ordered_df$X.Comment

pdf("/home/ida/master-thesis/results/parts_of_sequence.pdf")
ggplot(new_df_melt, aes(x = factor(peptide, level = peptide_order), y = value, fill = residues)) +
  geom_bar(position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Peptide") +
  ylab("AUC")
dev.off()

pdf("/home/ida/master-thesis/results/parts_of_sequence.pdf", height = 4)
ggplot(new_df_melt[new_df_melt$peptide != "Total (33048)" &
                     new_df_melt$peptide != "CLGGLLTMV (28)" &
                     new_df_melt$peptide != "ILKEPVHGV (16)",], 
       aes(x = factor(peptide, level = peptide_order), y = value, fill = residues)) +
  geom_bar(position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Peptide") +
  ylab("AUC") +
  scale_fill_manual(values = residue_colors, 
                    labels = c("All residues", "Without MHC", "Without peptide", "Without TCR"))
dev.off()

# Only total
pdf("/home/ida/master-thesis/results/parts_of_sequence_total.pdf", height = 3, width = 3)
ggplot(new_df_melt[new_df_melt$peptide == "Total (33048)",], 
       aes(x = residues, y = value)) +
  geom_bar(position = "dodge", stat = "identity", fill = "grey") +
  geom_text(aes(label=round(value, 2)), vjust=-0.25) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  xlab("Residues") +
  ylab("AUC") +
  ylim(0,0.9) +
  scale_x_discrete(labels=c("All residues", "Without MHC", "Without peptide", "Without TCR"))
dev.off()
