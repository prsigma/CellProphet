import os
import logging
import random
import numpy as np
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
import pandas as pd
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
from tqdm import tqdm
import scipy.io as sio

def change_into_current_py_path():
    current_file_path = __file__
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

def set_logger():
    logger = logging.getLogger(__name__)        
    logger.setLevel(logging.DEBUG)              
    console_handler = logging.StreamHandler()   
    logger.addHandler(console_handler)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - \33[32m%(message)s\033[0m')     
    console_handler.setFormatter(formatter) 
    logger.propagate = False
    return logger

def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_colors(N):
    """
    Generate N visually appealing colors using seaborn color palette.
    
    Args:
        N (int): The number of colors to generate.
    
    Returns:
        list: A list of N RGB tuples representing the generated colors.
    """
    color_palette = sns.color_palette("husl", N)
    colors = [mcolors.rgb2hex(color_palette[i]) for i in range(N)]
    return colors

def benchmark(gt_grn_path,pred_grn_list,algorithm_list,tf_list,outdir):
    gt_grn = pd.read_csv(gt_grn_path)
    gt_grn = gt_grn.drop_duplicates(keep='first', inplace=False).copy()
    gt_grn.reset_index(drop=True, inplace=True)
    gt_grn = gt_grn.loc[(gt_grn['Gene1'] != gt_grn['Gene2'])]

    # Load data
    colors = generate_colors(len(pred_grn_list))

    auc_list = list()
    auprc_list = list()
    f1_list = list()
    figure_roc,ax_roc =  plt.subplots()
    figure_auprc, ax_auprc=  plt.subplots()

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
        ax_auprc.plot(recall, precision, color=colors[i], label=algorithm_list[i])
        
        # Plot the ROC curve
        ax_roc.plot(fpr, tpr, color=colors[i], label=algorithm_list[i])

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

def computeScores(gt_grn_path, pred_grn_path):
    
    print("Calculating the AUROC AUPRC and F1...")

    gt_grn = pd.read_csv(gt_grn_path)
    gt_grn = gt_grn.drop_duplicates(keep='first', inplace=False).copy()
    gt_grn.reset_index(drop=True, inplace=True)
    gt_grn = gt_grn.loc[(gt_grn['Gene1'] != gt_grn['Gene2'])]

    pred_grn = pd.read_csv(pred_grn_path)
    pred_grn.drop(columns='Unnamed: 0',inplace=True)
    pred_grn.columns = ['Gene1','Gene2','weights_combined']
    pred_grn = pred_grn.sort_values('weights_combined', ascending=False)
    pred_grn = pred_grn.drop_duplicates(keep='first', inplace=False).copy()
    pred_grn.reset_index(drop=True, inplace=True)
    pred_grn = pred_grn.loc[(pred_grn['Gene1'] != pred_grn['Gene2'])]
    pred_grn = pred_grn[:len(gt_grn)]
    pred_grn.index = pred_grn['Gene1'] + "|" + pred_grn['Gene2']
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

    # Calculate the average precision score AUPR
    aupr = round(average_precision_score(result['label'], result['weights_combined']),3)
    
    return auc,aupr

def RGM_activity(network, pred_grn, expression_data, topk, num_workers: int = 8):
    """
    Select RGMs of driver regulators and calculate their activities in each cell.
    Activity score is calculated based on AUCell, which is from the `pyscenic` package.
    """
    from pyscenic.aucell import aucell
    from ctxcore.genesig import GeneSignature

    genes = pred_grn['Gene1'].unique()

    out_RGMs = dict()

    for i in genes:

        out_RGMs = {i:list(set(network.successors(i))) for i in genes
                    if len(set(network.successors(i))) >0}

    # n_cells x n_genes
    out_RGMs = [GeneSignature(name=k, gene2weight=v) for k, v in out_RGMs.items()]
    if len(out_RGMs) > 0:
        auc_mtx_out = aucell(expression_data, out_RGMs, num_workers=num_workers, normalize=True,noweights=True,auc_threshold=0.05, seed=2024)

    return auc_mtx_out

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
            # self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.best_model =  model.state_dict()
            # self.save_checkpoint(val_loss, model, path)
            

    # def save_checkpoint(self, val_loss, model):
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(, path)
    #     self.val_loss_min = val_loss