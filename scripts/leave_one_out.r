feature_set_colors <- brewer.pal(3, "Paired")

all_features <- read.csv("/home/ida/master-thesis/leave_one_out_allFeature_noMHC.csv")
all_features <- all_features[all_features$X.Comment != "CLGGLLTMV" &
                               all_features$X.Comment != "ILKEPVHGV" &
                               all_features$X.Comment != "Total",]
all_features$X.Comment <- paste(all_features$X.Comment, " (", all_features$count, ")", sep = "")

seq <- read.csv("/home/ida/master-thesis/leave_one_out_seq_noMHC.csv")
seq$X.Comment <- paste(seq$X.Comment, " (", seq$count, ")", sep = "")

energies <- read.csv("/home/ida/master-thesis/leave_one_out_energies_noMHC.csv")
energies$X.Comment <- paste(energies$X.Comment, " (", energies$count, ")", sep = "")

ggplot(all_features[all_features$X.Comment != "Total (33048)" &
                      all_features$X.Comment != "CLGGLLTMV (28)" &
                      all_features$X.Comment != "ILKEPVHGV (16)",], 
       aes(x = reorder(X.Comment, -count), y = AUC)) +
  geom_bar(position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Peptide")
  
all_named <- cbind(all_features, rep("all", 16))
seq_named <- cbind(seq, rep("seq", 16))
energies_named <- cbind(energies, rep("energies", 16))

colnames(all_named) <- c("Peptide", "AUC", "MCC", "Accuracy", "Precision", "Recall", 
                         "f1_score", "TN", "FP", "FN", "TP", "count", "feature_set")
colnames(seq_named) <- c("Peptide", "AUC", "MCC", "Accuracy", "Precision", "Recall", 
                         "f1_score", "TN", "FP", "FN", "TP", "count", "feature_set")
colnames(energies_named) <- c("Peptide", "AUC", "MCC", "Accuracy", "Precision", "Recall", 
                         "f1_score", "TN", "FP", "FN", "TP", "count", "feature_set")

all_data <- rbind(all_named, seq_named, energies_named)

pdf("/home/ida/master-thesis/results/leave_one_out_auc.pdf", height = 4)
ggplot(all_data, aes(x = reorder(Peptide, -count), y = AUC, fill = feature_set)) +
  geom_bar(position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Peptide")
dev.off()

pdf("/home/ida/master-thesis/results/leave_one_out_mcc.pdf", height = 4)
ggplot(all_data, aes(x = reorder(Peptide, -count), y = MCC, fill = feature_set)) +
  geom_bar(position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Peptide")
dev.off()