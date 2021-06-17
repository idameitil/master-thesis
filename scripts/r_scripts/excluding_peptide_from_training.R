
all_features <- read.csv("/home/ida/master-thesis/results/May25/allfeatures_noMHC_noYLLEMLWRL.csv")
seq <- read.csv("/home/ida/master-thesis/results/May25/onehot_noMHC_noYLLEMLWRL.csv")
energy_terms <- read.csv("/home/ida/master-thesis/results/May25/energyterms_noMHC_noYLLEMLWRL.csv")

data <- rbind(all_features[all_features$X.Comment == "YLLEMLWRL", "AUC"],
              seq[seq$X.Comment == "YLLEMLWRL", "AUC"], 
              energy_terms[energy_terms$X.Comment == "YLLEMLWRL", "AUC"])

data <- data.frame(c("all_features", "sequence", "energy_terms"), data)
colnames(data) <- c("feature_set", "AUC")

ggplot(data, aes(x = feature_set, y = AUC)) +
  geom_bar(stat = "identity")
