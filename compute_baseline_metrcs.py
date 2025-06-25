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

def EarlyPrec(trueEdgesDF: pd.DataFrame, predEdgesDF: pd.DataFrame,
              weight_key: str = 'weights_combined', TFEdges: bool = True):
    """
        Computes early precision for a given set of predictions in the form of a DataFrame.
        The early precision is defined as the fraction of true positives in the top-k edges,
        where k is the number of edges in the ground truth network (excluding self loops).

    :param trueEdgesDF:   A pandas dataframe containing the true edges.
    :type trueEdgesDF: DataFrame

    :param predEdgesDF:   A pandas dataframe containing the edges and their weights from
        the predicted network. Use param `weight_key` to assign the column name of edge weights.
        Higher the weight, higher the edge confidence.
    :type predEdgesDF: DataFrame

    :param weight_key:   A str represents the column name containing weights in predEdgeDF.
    :type weight_key: str

    :param TFEdges:   A flag to indicate whether to consider only edges going out of TFs (TFEdges = True)
        or not (TFEdges = False) from evaluation.
    :type TFEdges: bool

    :returns:
        - Eprec: Early precision value
        - Erec: Early recall value
        - EPR: Early precision ratio
    """

    print("Calculating the EPR(early prediction rate)...")

    # Remove self-loops
    trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
    if 'Score' in trueEdgesDF.columns:
        trueEdgesDF = trueEdgesDF.sort_values('Score', ascending=False)
    trueEdgesDF = trueEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    trueEdgesDF.reset_index(drop=True, inplace=True)

    predEdgesDF = predEdgesDF.loc[(predEdgesDF['Gene1'] != predEdgesDF['Gene2'])]
    if weight_key in predEdgesDF.columns:
        predEdgesDF = predEdgesDF.sort_values(weight_key, ascending=False)
    predEdgesDF = predEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    predEdgesDF.reset_index(drop=True, inplace=True)

    uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
    if TFEdges:
        # Consider only edges going out of source genes
        print("  Consider only edges going out of source genes")

        # Get a list of all possible TF to gene interactions
        possibleEdges_TF = set(product(set(trueEdgesDF.Gene1), set(uniqueNodes)))

        # Get a list of all possible interactions 
        possibleEdges_noSelf = set(permutations(uniqueNodes, r=2))

        # Find intersection of above lists to ignore self edges
        # TODO: is there a better way of doing this?
        possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)
        possibleEdges = pd.DataFrame(possibleEdges, columns=['Gene1', 'Gene2'], dtype=str)

        # possibleEdgesDict = {'|'.join(p): 0 for p in possibleEdges}
        possibleEdgesDict = possibleEdges['Gene1'] + "|" + possibleEdges['Gene2']

        trueEdges = trueEdgesDF['Gene1'].astype(str) + "|" + trueEdgesDF['Gene2'].astype(str)
        trueEdges = trueEdges[trueEdges.isin(possibleEdgesDict)]
        print("  {} TF Edges in ground-truth".format(len(trueEdges)))
        numEdges = len(trueEdges)

        predEdgesDF['Edges'] = predEdgesDF['Gene1'].astype(str) + "|" + predEdgesDF['Gene2'].astype(str)
        # limit the predicted edges to the genes that are in the ground truth
        predEdgesDF = predEdgesDF[predEdgesDF['Edges'].isin(possibleEdgesDict)]
        print("  {} Predicted TF edges are considered".format(len(predEdgesDF)))

        M = len(set(trueEdgesDF.Gene1)) * (len(uniqueNodes) - 1)

    else:
        trueEdges = trueEdgesDF['Gene1'].astype(str) + "|" + trueEdgesDF['Gene2'].astype(str)
        trueEdges = set(trueEdges.values)
        numEdges = len(trueEdges)
        print("  {} edges in ground-truth".format(len(trueEdges)))

        M = len(uniqueNodes) * (len(uniqueNodes) - 1)

    if not predEdgesDF.shape[0] == 0:
        # Use num True edges or the number of
        # edges in the dataframe, which ever is lower
        maxk = min(predEdgesDF.shape[0], numEdges)
        edgeWeightTopk = predEdgesDF.iloc[maxk - 1][weight_key]

        nonZeroMin = np.nanmin(predEdgesDF[weight_key].values)
        bestVal = max(nonZeroMin, edgeWeightTopk)

        newDF = predEdgesDF.loc[(predEdgesDF[weight_key] >= bestVal)]
        predEdges = set(newDF['Gene1'].astype(str) + "|" + newDF['Gene2'].astype(str))
        print("  {} Top-k edges selected".format(len(predEdges)))
    else:
        predEdges = set([])

    if len(predEdges) != 0:
        intersectionSet = predEdges.intersection(trueEdges)
        print("  {} true-positive edges".format(len(intersectionSet)))
        Eprec = len(intersectionSet) / len(predEdges)
        Erec = len(intersectionSet) / len(trueEdges)
    else:
        Eprec = 0
        Erec = 0

    random_EP = len(trueEdges) / M
    EPR = Erec / random_EP
    return Eprec, Erec, EPR

def benchmark(gt_grn_path,pred_grn_list,algorithm_list,tf_list,outdir,each_dataset):
    gt_grn = pd.read_csv(gt_grn_path)
    gt_grn = gt_grn.drop_duplicates(keep='first', inplace=False).copy()
    gt_grn.reset_index(drop=True, inplace=True)
    gt_grn = gt_grn.loc[(gt_grn['Gene1'] != gt_grn['Gene2'])]

    # Load data
    colors = ['#d45740', '#8790b1', '#6cb8d1', '#e79f85', '#425385', '#479e88', '#9ed0c2','#9ed1c3', '#f4a261', '#2a9d8f']  # Colors inspired by Matplotlib tableau palette

    auc_list = list()
    auprc_list = list()
    f1_list = list()
    epr_list = list()
    figure_roc,ax_roc =  plt.subplots(figsize=(5, 5))
    figure_auprc, ax_auprc=  plt.subplots(figsize=(5, 5))

    uniqueNodes = np.unique(gt_grn.loc[:, ['Gene1', 'Gene2']])
    trueEdges = set(gt_grn['Gene1'].astype(str) + "|" + gt_grn['Gene2'].astype(str))
    numEdges = len(trueEdges)
    M = len(set(gt_grn['Gene1'])) * (len(uniqueNodes) - 1)  # TF edges case

    for i in tqdm(range(len(pred_grn_list))):
        print(each_dataset)
        print(algorithm_list[i])
        if algorithm_list[i] == 'CEFCON':
            pred_grn = pd.read_csv(pred_grn_list[i],header=None)
        else:
            pred_grn = pd.read_csv(pred_grn_list[i])
        if 'Unnamed: 0' in pred_grn.columns:
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
        
        if not pred_grn.shape[0] == 0:
            maxk = min(pred_grn.shape[0], numEdges)
            if maxk > 0:
                edgeWeightTopk = pred_grn.iloc[maxk - 1]['weights_combined']
                nonZeroMin = np.nanmin(pred_grn['weights_combined'].values)
                bestVal = max(nonZeroMin, edgeWeightTopk)
                
                newDF = pred_grn.loc[(pred_grn['weights_combined'] >= bestVal)]
                predEdges = set(newDF['index'])
                
                if len(predEdges) != 0:
                    intersectionSet = predEdges.intersection(trueEdges)
                    Eprec = len(intersectionSet) / len(predEdges)
                    Erec = len(intersectionSet) / len(trueEdges)
                    random_EP = len(trueEdges) / M
                    EPR = Erec / random_EP if random_EP > 0 else 0
                else:
                    EPR = 0
            else:
                EPR = 0
        else:
            EPR = 0
        
        epr_list.append({'EPR': round(EPR, 3)})
        
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
    figure_auprc.savefig(os.path.join(outdir,"pr_curve.svg"), bbox_inches='tight')
    plt.close(figure_auprc)

    ax_roc.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.0])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc="lower right")
    figure_roc.savefig(os.path.join(outdir,"roc_curve.svg"), bbox_inches='tight')
    plt.close(figure_roc)
        
    auc_df = pd.DataFrame(auc_list)
    auprc_df = pd.DataFrame(auprc_list)
    f1_df = pd.DataFrame(f1_list)
    epr_df = pd.DataFrame(epr_list)
    metric_df = pd.concat([auc_df,auprc_df],axis=1)
    metric_df = pd.concat([metric_df,f1_df],axis=1)
    metric_df = pd.concat([metric_df,epr_df],axis=1)
    
    return metric_df

datasets = ["mESC","mHSC-L","mHSC-E","mHSC-GM","hESC","hHep","mDC"]

root_path = "output"

algorithm = ['TRIGON','CEFCON','Celloracle','NetREX','GRNBoost2','GENIE3','Random',"Prior_Random","GRANGER","Inferelator"]


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
            
        elif each_algorithm == 'Celloracle' or each_algorithm == 'GRNBoost2' or each_algorithm == 'Random' or each_algorithm == 'GENIE3' or each_algorithm == 'TRIGON' or each_algorithm == 'Prior_Random' or each_algorithm == 'GRANGER' or each_algorithm == 'Inferelator':
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
    
    metric_df = benchmark(gt_grn_path,pred_grn_list,algorithm_list,tf_list,f'{root_path}/{each_dataset}/',each_dataset)
    metric_df['dataset'] = each_dataset
    metric_list.append(metric_df)

whole_metric_df = pd.concat(metric_list,axis=0)
whole_metric_df.to_csv(f'{root_path}/whole_metric.csv',index=False)
        