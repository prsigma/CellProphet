library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
gene_ensembls_path <- args[1]
output_path <- args[2]

gene_ensembls <- read.table(gene_ensembls_path ,header=F)
genes <- as.character(gene_ensembls$V1)
ego <- enrichGO(gene=genes, OrgDb=org.Hs.eg.db, keyType='ENSEMBL', ont="BP", pAdjustMethod="BH", pvalueCutoff=0.05, qvalueCutoff=0.2, readable=T)
write.table(as.data.frame(ego),output_path + "/go_enrich.csv",sep="\t",row.names =F,quote=F)