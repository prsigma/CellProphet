library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)
args <- commandArgs(trailingOnly = TRUE)
gene_names_path <- args[1]
output_path <- args[2]
gene_names <- read.table(gene_names_path, header=F)
gene_ensembls = bitr(gene_names$V1, fromType="SYMBOL", toType=c("ENSEMBL", "ENTREZID"), OrgDb=org.Hs.eg.db)
write.csv(gene_ensembls, file=output_path + "/cluster_infos.csv", row.names=F)