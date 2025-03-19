from tqdm import tqdm
import pandas as pd
import os
import logging
import random
import numpy as np
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import sparse
import networkx as nx
import cvxpy as cvx
import seaborn as sns
import matplotlib.ticker as ticker
from itertools import product, permutations
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score, f1_score
from matplotlib.pyplot import rc_context
from matplotlib.pyplot import MultipleLocator
import matplotlib.colors as mcolors
import scipy.io as sio

import matplotlib.font_manager as fm

# plt.rc('font', family='Arial')  # Set font to Times New Roman

def benchmark(gt_grn_path,pred_grn_list,algorithm_list,tf_list,outdir):
    gt_grn = pd.read_csv(gt_grn_path)
    gt_grn = gt_grn.drop_duplicates(keep='first', inplace=False).copy()
    gt_grn.reset_index(drop=True, inplace=True)
    gt_grn = gt_grn.loc[(gt_grn['Gene1'] != gt_grn['Gene2'])]

    # Load data
    colors = ['#d45740', '#8790b1', '#6cb8d1', '#e79f85', '#425385', '#479e88', '#9ed0c2','#9ed1c3']  # Colors inspired by Matplotlib tableau palette

    auc_list = list()
    auprc_list = list()
    f1_list = list()
    figure_roc,ax_roc =  plt.subplots(figsize=(5, 5))
    figure_auprc, ax_auprc=  plt.subplots(figsize=(5, 5))

    for i in tqdm(range(len(pred_grn_list))):
        if i == 1:
            pred_grn = pd.read_csv(pred_grn_list[i],header=None)
        else:
            pred_grn = pd.read_csv(pred_grn_list[i])
        if len(pred_grn.columns) == 3:
            pass
        else:
            pred_grn.drop(columns=['Unnamed: 0'],inplace=True)
        pred_grn.columns = ['Gene1','Gene2','weights_combined']
        pred_grn = pred_grn.sort_values('weights_combined', ascending=False)
        pred_grn = pred_grn.drop_duplicates(keep='first', inplace=False).copy()
        pred_grn.reset_index(drop=True, inplace=True)
        pred_grn = pred_grn.loc[(pred_grn['Gene1'] != pred_grn['Gene2'])]
        pred_grn = pred_grn[:len(gt_grn)]
        pred_grn.index = pred_grn['Gene1'] + "|" + pred_grn['Gene2']
        pred_grn = pred_grn[pred_grn['Gene1'].isin(tf_list)]
        pred_grn = pred_grn.reset_index()

        possibleEdges = list(permutations(np.unique(gt_grn.loc[:, ['Gene1', 'Gene2']]),
                                        r=2))
        
        TrueEdgeDict = pd.DataFrame({'|'.join(p): 0 for p in possibleEdges}, index=['label']).T
        TrueEdgeDict.loc[np.array(gt_grn['Gene1'] + "|" + gt_grn['Gene2']), 'label'] = 1
        TrueEdgeDict = TrueEdgeDict.reset_index()

        result = pd.merge(pred_grn,TrueEdgeDict,on='index',how='inner')

        fpr, tpr, _ = roc_curve(y_true=result['label'], y_score=result['weights_combined'],pos_label=1)
        
        # Compute AUC
        auc = round(roc_auc_score(result['label'], result['weights_combined']),3)
        auc_list.append({'Algorithm':algorithm_list[i],'AUROC':auc})

        # Calculate the average precision score AUPR
        aupr = round(average_precision_score(result['label'], result['weights_combined']),3)
        auprc_list.append({'AUPRC':aupr})

        # calculate f1
        result['predict'] = 1
        f1 = round(f1_score(y_true=result['label'], y_pred=result['predict']),3)
        f1_list.append({'f1':f1})
        
        # Calculate precision and recall values
        precision, recall, _ = precision_recall_curve(y_true=result['label'], probas_pred=result['weights_combined'], pos_label=1)
        
        # Plot precision-recall curve
        ax_auprc.plot(recall, precision, color=colors[i], label=f"{algorithm_list[i]}(AUPRC={aupr})")
        
        # Plot the ROC curve
        ax_roc.plot(fpr, tpr, color=colors[i], label=f"{algorithm_list[i]}(AUROC={auc})")

    ax_auprc.set_xlim([0.0, 1.0])
    ax_auprc.set_ylim([0.0, 1.0])
    ax_auprc.set_xlabel('Recall')
    ax_auprc.set_ylabel('Precision')
    ax_auprc.set_title('Precision-Recall Curve')
    ax_auprc.legend(loc='upper right')
    figure_auprc.savefig(os.path.join(outdir,"pr_curve.png"), bbox_inches='tight')
    plt.close(figure_auprc)

    ax_roc.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.0])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc="lower right")
    figure_roc.savefig(os.path.join(outdir,"roc_curve.png"), bbox_inches='tight')
    plt.close(figure_roc)
        
    auc_df = pd.DataFrame(auc_list)
    auprc_df = pd.DataFrame(auprc_list)
    f1_df = pd.DataFrame(f1_list)
    metric_df = pd.concat([auc_df,auprc_df],axis=1)
    metric_df = pd.concat([metric_df,f1_df],axis=1)
    
    return metric_df

datasets = ["mESC","mHSC-GM","mHSC-L","mHSC-E"]
# datasets = ['mESC']

root_path = "output"

algorithm = ['TRIGON','CEFCON','Celloracle','NetREX','GRNBoost2','GENIE3','Random',"Prior_Random"]

metric_list = list()
for each_dataset in datasets:
    pred_grn_list = list()
    algorithm_list = list()
    gt_grn_path = f'{root_path}/{each_dataset}/gt_grn.csv'
    tf_path = f'{root_path}/{each_dataset}/end_tf.csv'
    tf_list = pd.read_csv(tf_path)['tf'].unique()

    for each_algorithm in tqdm(algorithm):
        path = f'{root_path}/{each_dataset}/{each_algorithm}/'
        algorithm_list.append(each_algorithm)
        
        if each_algorithm == 'CEFCON':
            pred_grn_list.append(os.path.join(path,'cell_lineage_GRN.csv'))
            
        elif each_algorithm == 'Celloracle' or each_algorithm == 'GRNBoost2' or each_algorithm == 'Random' or each_algorithm == 'GENIE3' or each_algorithm == 'TRIGON' or each_algorithm == 'Prior_Random':
            pred_grn_list.append(os.path.join(path,'grn.csv'))
        
        elif each_algorithm == 'NetREX':
            pred_grn = pd.read_csv(os.path.join(path,'NetREX_PredictedNetwork.tsv'),sep='\t')
            pred_grn = pred_grn.melt(id_vars=['Unnamed: 0'])
            pred_grn.columns = ['TF','Gene','weights_combined']
            pred_grn['TF'] = pred_grn['TF'].map(lambda x:x.upper())
            pred_grn['Gene'] = pred_grn['Gene'].map(lambda x:x.upper())

            grn_path = os.path.join(path,'grn.csv')
            pred_grn.to_csv(grn_path)
            pred_grn_list.append(grn_path)
            
        else:
            pass
    
    metric_df = benchmark(gt_grn_path,pred_grn_list,algorithm_list,tf_list,f'{root_path}/{each_dataset}/')
    metric_df['dataset'] = each_dataset
    metric_list.append(metric_df)

whole_metric_df = pd.concat(metric_list,axis=0)
whole_metric_df.to_csv(f'{root_path}/whole_metric.csv',index=False)
        