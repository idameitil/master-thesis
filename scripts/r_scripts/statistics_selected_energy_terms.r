
selected_energy_terms <- readRDS(file = "selected_energy_terms.rds")
colnames(selected_energy_terms)[14] <- "origin"
melt_df <- melt(selected_energy_terms)
colnames(melt_df) <- c("origin", "features", "value")

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

labs = c("foldx_MP" = "MHC-peptide",
                        "foldx_MA" = "MHC-TCRa",
                        "foldx_MB" = "MHC-TCRb",
                        "foldx_PA" = "peptide-TCRa",
                        "foldx_PB" = "peptide-TCRb",
                        "foldx_AB" = "TCRa-TCRb")
#labs <- c("MHC-peptide","MHC-TCRa","MHC-TCRb","peptide-TCRa","peptide-TCRb","TCRa-TCRb")

pdf("/home/ida/master-thesis/results/boxplots_foldX.pdf", width = 7, height = 4)
ggplot(melt_df[melt_df$features=="foldx_MP" |
                 melt_df$features=="foldx_MA" |
                 melt_df$features=="foldx_MB" |
                 melt_df$features=="foldx_PA" |
                 melt_df$features=="foldx_PB" |
                 melt_df$features=="foldx_AB",], 
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  geom_jitter(color="snow3", size=0.0001, alpha=0.1) +
  ylab("") +
  xlab("") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped\nnegative", tenx = "10X\nnegative")) +
  facet_wrap(~ features, scales = "free", ncol = 3, labeller = as_labeller(labs))
dev.off()

pdf("/home/ida/master-thesis/results/boxplots_energies_one.pdf", width = 10, height = 14)
ggplot(melt_df,
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  #theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  geom_jitter(color="snow3", size=0.0001, alpha=0.3) +
  ylab("") +
  xlab("") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped neg", tenx = "10X neg")) +
  facet_wrap(~ features, scales = "free", ncol = 3)
dev.off()
  

### FOR EACH ROSETTA ###
pdf("/home/ida/master-thesis/results/boxplot_total.pdf", width = 4, height = 4)
a <- ggplot(melt_df[melt_df$features=="delta_total",], 
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  geom_jitter(color="snow3", size=0.0001, alpha=0.1) +
  ylab("") +
  xlab("") +
  ggtitle("total") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped\nnegative", tenx = "10X\nnegative")) +
  #scale_y_continuous(trans='log10') +
  scale_y_continuous()
dev.off()

pdf("/home/ida/master-thesis/results/boxplot_fa_atr.pdf", width = 4, height = 4)
b <- ggplot(melt_df[melt_df$features=="delta_fa_atr",], 
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  geom_jitter(color="snow3", size=0.0001, alpha=0.1) +
  ylab("") +
  xlab("") +
  ggtitle("fa_atr") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped\nnegative", tenx = "10X\nnegative")) +
  #scale_y_continuous(trans='log10') +
  scale_y_continuous(limits = quantile(melt_df[melt_df$features=="delta_fa_atr","value"], c(0.01, 0.99)))
dev.off()

pdf("/home/ida/master-thesis/results/boxplot_fa_rep.pdf", width = 4, height = 4)
c <- ggplot(melt_df[melt_df$features=="delta_fa_rep",], 
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  geom_jitter(color="snow3", size=0.0001, alpha=0.1) +
  ylab("") +
  xlab("") +
  ggtitle("fa_rep") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped\nnegative", tenx = "10X\nnegative")) +
  #scale_y_continuous(trans='log10') +
  scale_y_continuous(limits = quantile(melt_df[melt_df$features=="delta_fa_rep","value"], c(0.01, 0.99)))
dev.off()

pdf("/home/ida/master-thesis/results/boxplot_fa_sol.pdf", width = 4, height = 4)
d <- ggplot(melt_df[melt_df$features=="delta_fa_sol",], 
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  geom_jitter(color="snow3", size=0.0001, alpha=0.1) +
  ylab("") +
  xlab("") +
  ggtitle("fa_sol") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped\nnegative", tenx = "10X\nnegative")) +
  #scale_y_continuous(trans='log10') +
  scale_y_continuous(limits = quantile(melt_df[melt_df$features=="delta_fa_sol","value"], c(0.01, 0.99)))
dev.off()

pdf("/home/ida/master-thesis/results/boxplot_fa_elec.pdf", width = 4, height = 4)
e <- ggplot(melt_df[melt_df$features=="delta_fa_elec",], 
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  geom_jitter(color="snow3", size=0.0001, alpha=0.1) +
  ylab("") +
  xlab("") +
  ggtitle("fa_elec") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped\nnegative", tenx = "10X\nnegative")) +
  #scale_y_continuous(trans='log10') +
  scale_y_continuous(limits = quantile(melt_df[melt_df$features=="delta_fa_elec","value"], c(0.01, 0.99)))
dev.off()

pdf("/home/ida/master-thesis/results/boxplot_fa_dun.pdf", width = 4, height = 4)
f <- ggplot(melt_df[melt_df$features=="delta_fa_dun",], 
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  geom_jitter(color="snow3", size=0.0001, alpha=0.1) +
  ylab("") +
  xlab("") +
  ggtitle("fa_dun") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped\nnegative", tenx = "10X\nnegative")) +
  #scale_y_continuous(trans='log10') +
  scale_y_continuous(limits = quantile(melt_df[melt_df$features=="delta_fa_dun","value"], c(0.01, 0.99)))
dev.off()

pdf("/home/ida/master-thesis/results/boxplot_p_aa_pp.pdf", width = 4, height = 4)
g <- ggplot(melt_df[melt_df$features=="delta_p_aa_pp",], 
       aes(x = origin, y = value, fill = origin)) +
  geom_boxplot() +
  geom_jitter(color="snow3", size=0.0001, alpha=0.1) +
  ylab("") +
  xlab("") +
  ggtitle("p_aa_pp") +
  theme(legend.position = "none") +
  scale_x_discrete(labels=c(pos = "Positive", swap = "Swapped\nnegative", tenx = "10X\nnegative")) +
  #scale_y_continuous(trans='log10') +
  scale_y_continuous(limits = quantile(melt_df[melt_df$features=="delta_p_aa_pp","value"], c(0.01, 0.99)))
dev.off()

library('gridExtra')
pdf("/home/ida/master-thesis/results/boxplots_rosetta.pdf", width = 7, height = 4)
grid.arrange(a, b, c, d, e, f, nrow = 2) 
dev.off()

# ANOVA
res.aov <- aov(value ~ origin, data = melt_df[melt_df$feature=="foldx_PB",])
summary(res.aov)
TukeyHSD(res.aov)
coefficients(res.aov)
plot(res.aov, 2)