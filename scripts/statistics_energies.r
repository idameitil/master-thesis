library(RcppCNPy)
library(stringr)
library(dplyr)
library(reshape2)
library(ggplot2)

### READ DATA ###
data_dir = "/home/ida/master-thesis/data/train_data/"
file_list = list.files(path = data_dir)

global_features_df <- data.frame(matrix(ncol = 79, nrow = length(file_list)))
colnames(global_features_df) <- c("foldx_MP","foldx_MA","foldx_MB","foldx_PA","foldx_PB",
                                  "foldx_AB","complex_total_score","complex_score",	
                                  "complex_dslf_fa13","complex_fa_atr","complex_fa_dun",
                                  "complex_fa_elec","complex_fa_intra_rep",
                                  "complex_fa_intra_sol_xover4","complex_fa_rep",
                                  "complex_fa_sol","complex_hbond_bb_sc",
                                  "complex_hbond_lr_bb","complex_hbond_sc",
                                  "complex_hbond_sr_bb","complex_linear_chainbreak",
                                  "complex_lk_ball_wtd","complex_omega",
                                  "complex_overlap_chainbreak","complex_p_aa_pp",
                                  "complex_pro_close","complex_rama_prepro",
                                  "complex_ref","complex_time","complex_yhh_planarity",
                                  "tcr_total_score","tcr_score","tcr_dslf_fa13","tcr_fa_atr",
                                  "tcr_fa_dun","tcr_fa_elec","tcr_fa_intra_rep",
                                  "tcr_fa_intra_sol_xover4","tcr_fa_rep","tcr_fa_sol",
                                  "tcr_hbond_bb_sc","tcr_hbond_lr_bb","tcr_hbond_sc",
                                  "tcr_hbond_sr_bb","tcr_linear_chainbreak","tcr_lk_ball_wtd",
                                  "tcr_omega","tcr_overlap_chainbreak","tcr_p_aa_pp",
                                  "tcr_pro_close","tcr_rama_prepro","tcr_ref","tcr_time",
                                  "tcr_yhh_planarity","pmhc_total_score","pmhc_score",
                                  "pmhc_dslf_fa13","pmhc_fa_atr","pmhc_fa_dun","pmhc_fa_elec",
                                  "pmhc_fa_intra_rep","pmhc_fa_intra_sol_xover4",
                                  "pmhc_fa_rep","pmhc_fa_sol","pmhc_hbond_bb_sc",
                                  "pmhc_hbond_lr_bb","pmhc_hbond_sc","pmhc_hbond_sr_bb",
                                  "pmhc_linear_chainbreak","pmhc_lk_ball_wtd","pmhc_omega",
                                  "pmhc_overlap_chainbreak","pmhc_p_aa_pp","pmhc_pro_close",
                                  "pmhc_rama_prepro","pmhc_ref","pmhc_time",
                                  "pmhc_yhh_planarity","group")

rownames_vector <- c()
for (i in sequence(length(file_list)))
{
  full_mat <- npyLoad(paste(data_dir, file_list[i], sep=""))
  #full_mat <- mutate_all(full_mat, function(x) as.numeric(as.character(x)))
  global_features <- full_mat[1,-(1:64)]
  id = sub("\\_.*", "", file_list[i])
  rownames_vector = c(rownames_vector, id)
  if (str_detect(file_list[i], "pos")){
    group = "pos"
  }
  if (str_detect(file_list[i], "tenx")){
    group = "tenx"
  }
  if (str_detect(file_list[i], "swap")){
    group = "swap"
  }
  global_features_df[i,] <- c(global_features, group)
}
rownames(global_features_df) <- rownames_vector

global_features_numeric <- mutate_all(global_features_df[,-79], function(x) as.numeric(as.character(x)))

global_features_final <- cbind(global_features_numeric, factor(global_features_df[,- (1:78)]))
summary(global_features_final)

meltData <- melt(global_features_final)
colnames(meltData) <- c("group", "feature", "value")

saveRDS(meltData, file = "global_energies.rds")

### MAIN ###
# Read data from rds file
meltData2 <- readRDS(file = "global_energies.rds")

# summary
summary(meltData)

# Make boxplots all features
for (feature in unique(meltData$feature)){
  pdf(paste("/home/ida/master-thesis/results/boxplots/", feature, ".pdf", sep=""))
  print(ggplot(meltData[meltData$feature==feature,], aes(x = group, y = value, fill = group)) +
    geom_boxplot() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    geom_jitter(color="darkgrey", size=0.001, alpha=0.9) +
    ggtitle(feature))
  dev.off()
}

# Complex total score group means and sd
group_by(meltData[meltData$feature=="complex_total_score",], group) %>%
  summarise(
    count = n(),
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE)
  )

# Complex total score ANOVA
res.aov <- aov(value ~ group, data = meltData[meltData$feature=="complex_total_score",])
summary(res.aov)
TukeyHSD(res.aov)
coefficients(res.aov)
plot(res.aov, 2)
lm.model <- lm(value ~ group, data = meltData[meltData$feature=="complex_total_score",])
summary(lm.model)
