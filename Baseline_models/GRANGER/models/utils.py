import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import *
import os
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import average_precision_score
def get_tf_list(tf_path):
    f_tf = open(tf_path)
    tf_reader = list(csv.reader(f_tf)) 
    tf_list = []
    for single in tf_reader[1:]:
        tf_list.append(single[0])
    # print('Load ' + str(len(tf_list)) + ' TFs successfully!')
    return tf_list      

def get_origin_expression_data(gene_expression_path):
    f_expression = open(gene_expression_path, encoding="utf-8")
    expression_reader = list(csv.reader(f_expression))
    cells = expression_reader[0][1:]
    num_cells = len(cells)
    expression_record = {}
    num_genes = 0
    for single_expression_reader in expression_reader[1:]:
        if single_expression_reader[0] in expression_record:
            print('Gene name ' + single_expression_reader[0] + ' repeat!')
        expression_record[single_expression_reader[0]] = list(map(float, single_expression_reader[1:]))
        num_genes += 1
    print(str(num_genes) + ' genes and ' + str(num_cells) + ' cells are included in origin expression data.')
    f_expression.close()
    return expression_record, cells 

def get_normalized_expression_data(gene_expression_path):
    expression_record, cells = get_origin_expression_data(gene_expression_path)
    expression_matrix = np.zeros((len(expression_record), len(cells)))
    index_row = 0
    for gene in expression_record:
        expression_record[gene] = np.log10(np.array(expression_record[gene]) + 10 ** -2)
        expression_matrix[index_row] = expression_record[gene]
        index_row += 1
    return expression_record, cells 

def get_lable_dic(gene_pair_list_path): 
    f_genePairList = open(gene_pair_list_path, encoding='UTF-8')  
    label_list=[]
    for single_pair in list(csv.reader(f_genePairList))[1:]:
         label_list.append(single_pair)
    f_genePairList.close()
    label_list_update = [[1 if item == '+' else 0 if item == '-' else item for item in sublist] for sublist in label_list]
    f_genePairList.close()
    return label_list_update
    
def to_causal_arr(casual_martix_path,gene_expression_path):
    #store the corresponding gene names and probabilities of causal relationships in a list
    matrix = np.genfromtxt(casual_martix_path, delimiter=',').T
    genes = get_tf_list(gene_expression_path)
    causal_list = []
    max_prob = np.max(matrix) 
    for row in range(matrix.shape[0]): 
        for col in range(matrix.shape[1]):  
            gene1 = genes[row]
            gene2 = genes[col]
            causal_prob=matrix[row,col]
            causal_prob_normalized = causal_prob / max_prob
            causal_list.append([gene1, gene2, causal_prob_normalized])
    # print(causal_list)
    return causal_list


def filtered_predict_pair(label_pair,predict_pair):
    #filter predict pair to compare with label_pair
    label_dict = {tuple(pair[:2]): idx for idx, pair in enumerate(label_pair)}  
    filtered_predict_pair = [pair for pair in predict_pair if tuple(pair[:2]) in label_dict]  
    sorted_predict_pair = [None] * len(label_pair)  
    for idx, (key, _) in enumerate(label_dict.items()):  
        for predict_pair_item in filtered_predict_pair:  
            if tuple(predict_pair_item[:2]) == key:  
                sorted_predict_pair[idx] = predict_pair_item  
                break    
    # print(sorted_predict_pair)
    labels = [pair[2] for pair in label_pair]  
    predictions = [pair[2] for pair in sorted_predict_pair]  
    # acc AUROC  
    auroc = roc_auc_score(labels, predictions)  
    aupr = average_precision_score(labels, predictions) 
    print(f"AUROC: {auroc}")
    print(f"AUPR:{aupr}")
    return sorted_predict_pair,auroc,aupr





