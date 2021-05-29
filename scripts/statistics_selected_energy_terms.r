
selected_energy_terms <- readRDS(file = "selected_energy_terms.rds")
melt_df <- melt(selected_energy_terms)
colnames(melt_df) <- c("origin", "features", "value")

for (feature in unique(melt_df$feature)){
  pdf(paste("/home/ida/master-thesis/results/boxplots_selected/", feature, ".pdf", sep=""), width = 5, height = 5)
  print(ggplot(melt_df[melt_df$feature==feature,], aes(x = origin, y = value, fill = origin)) +
          geom_boxplot() +
          #theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
          geom_jitter(color="darkgrey", size=0.001, alpha=0.9) +
          xlab("") +
          theme(legend.position = "none") +
          scale_x_discrete(labels=c(pos = "Positives", swap = "Swapped negatives", tenx = "10X negatives"))
  )
  dev.off()
}

melt_df$features

pdf("/home/ida/master-thesis/results/boxplots_energies_one.pdf", width = 10, height = 14)
ggplot(melt_df, aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  #theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  geom_jitter(color="snow3", size=0.0001, alpha=0.3) +
  ylab("") +
  xlab("") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped neg", tenx = "10X neg")) +
  facet_wrap(~ features, scales = "free", ncol = 3)
dev.off()
  