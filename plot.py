import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *
from tqdm import tqdm

current_path = os.path.dirname(__file__)
os.chdir(current_path)

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

path = ('./Output/output_1000_specific/',)

def plot_gene_trends(gene_trends, genes, color, figsize=None):
    """ 
    :param: gene_trends: Results of the compute_marker_trends function
    """
    sns.set_theme(style="ticks")
    # Branches and genes
    branches = list(gene_trends.keys())
    if genes is None:
        genes = gene_trends[branches[0]]["trends"].index

    # Set up figure
    if figsize is None:
        figsize = len(genes), 13
    fig, axes = plt.subplots(figsize=figsize, nrows=len(genes), sharex='col')
    axes = np.array(axes).reshape((-1,))
    fig.tight_layout()
    
    for i, (gene, ax) in enumerate(zip(genes, axes)):
        for branch in branches:
            trends = gene_trends[branch]["trends"]
            stds = gene_trends[branch]["std"]
            ax.plot(
                trends.columns, trends.loc[gene, :], color=color, label=branch
            )
            ax.fill_between(
                trends.columns,
                trends.loc[gene, :] - stds.loc[gene, :],
                trends.loc[gene, :] + stds.loc[gene, :],
                alpha=0.1,
                color=color,
            )
            ax.set_ylabel(gene)

    ax.set_xticks([0, 1])
    sns.despine()
    
    return fig

def plot_aucell():
    change_into_current_py_path() 
    dataset = ["mESC"]

    for each_dataset in dataset:
        root_path = './output'
        each_alg = 'MTGRN'

        path = f'{root_path}/{each_dataset}/{each_alg}/'

        auc_mtx_out = pd.read_csv('/home/pengrui/STGRN/aucell.csv',sep=',')
        auc_mtx_out = auc_mtx_out.set_index('Rowname')
        
        
        cell_sort_path = f'/home/pengrui/STGRN/output/{each_dataset}/MTGRN/mtgrn_cell_sort.csv'
        cell_sort_df = pd.read_csv(cell_sort_path)

        
        if each_dataset == "hHep":
            cell_sort_df['true_time'] = cell_sort_df['cell'].map(lambda x:x.split('_')[1])
            for i,time in enumerate(['ipsc','de','he1','he2','ih1','mh1','mh2']):
                cell_sort_df.loc[cell_sort_df['true_time'] == time, 'time_label'] = i
        
        elif each_dataset == "mESC":
            cell_sort_df['true_time'] = cell_sort_df['cell'].map(lambda x:x.split('_')[2])
            for i,time in enumerate(['00h','12h','24h','48h','72h']):
                cell_sort_df.loc[cell_sort_df['true_time'] == time, 'time_label'] = i

        else:
            cell_sort_df['true_time'] = cell_sort_df['cell'].map(lambda x: '_'.join(x.split("_")[:-1]))
            for i,time in enumerate(['LT_HSC','HSPC','Prog']):
                cell_sort_df.loc[cell_sort_df['true_time'] == time, 'time_label'] = i
        
        auc_mtx_out = auc_mtx_out[cell_sort_df['cell']]

        time_label =  cell_sort_df['time_label'].tolist()

        result = master_regulator(auc_mtx_out,cell_sort_df)
        result = result[result['adj_p'] < 0.01]
        auc_mtx_out = auc_mtx_out.loc[result.index]
        
        save_path = os.path.join(root_path,'RGM_results')
        os.makedirs(save_path,exist_ok=True)

        plot_RGM_activity_heatmap(auc_mtx_out,time_label,os.path.join(save_path,f'{each_dataset}_RGM.png'))
        result.to_csv(os.path.join(path,f't_test_driver_gene.csv'))

if __name__ == '__main__':
    plot_aucell()
    
        




    