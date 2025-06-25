import csv  
import random  
import pandas as pd
import numpy as np
from scipy.stats import spearmanr 
def get_tf_list(tf_path):  
    with open(tf_path, newline='', encoding='utf-8') as f_tf:  
        tf_reader = csv.reader(f_tf)    
        next(tf_reader)  
        tf_list = [row[0] for row in tf_reader]  
    return tf_list   
  
def spear(gene_a,gene_b,file_path):  
    df = pd.read_csv(file_path, index_col=0) 
    #Calculate the Spearman correlation between genes
    gene_a = df.loc[gene_a, :]  
    gene_b = df.loc[gene_b, :] 
    corr, _ = spearmanr(gene_a, gene_b)
    if corr>0:
        sign='+'
    else:
        sign='-'
    return sign

def to_causal_arr(casual_martix_path,gene_expression_path):
    matrix = np.genfromtxt(casual_martix_path, delimiter=',').T
    matrix=matrix.T
    max_prob = np.max(matrix) 
    average = np.mean(matrix)
    min_prob=np.min(matrix)
    genes = get_tf_list(gene_expression_path)
    causal_list = []
    max_prob = np.max(matrix) 
    for row in range(matrix.shape[0]): 
        for col in range(matrix.shape[1]):  
            gene1 = genes[row]
            gene2 = genes[col]
            causal_prob=matrix[row,col]
            sign=spear(gene1,gene2,gene_expression_path)
            causal_prob_normalized = (causal_prob-min_prob) / (max_prob-min_prob)
            if causal_prob>average:
                causal_prob=1
            else:
                causal_prob=0
            causal_list.append([gene1, gene2, causal_prob,sign,causal_prob_normalized])

    return causal_list

# # example
a='GC_cell.csv'
b= 'example data/mCAD-2000-1/ExpressionData.csv'  # your CSV file path 
c=to_causal_arr(a,b)
# standard format(genea geneb correlation score)
with open('result.csv', 'w', newline='') as csvfile:  
    writer = csv.writer(csvfile)   
    for row in c:  
        writer.writerow(row)   
print("The format result CSV file is ready!")

