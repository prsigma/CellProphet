import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *
from tqdm import tqdm

current_path = os.path.dirname(__file__)
os.chdir(current_path)

sns.set_theme(font='Times New Roman',font_scale=1.4)
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
        
        #读取细胞顺序
        cell_sort_path = f'/home/pengrui/STGRN/output/{each_dataset}/MTGRN/mtgrn_cell_sort.csv'
        cell_sort_df = pd.read_csv(cell_sort_path)

        #寻找每一个阶段细胞的数量
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
    
    # """
    # plot gene trend
    # """
    # import scanpy as sc
    # import scanpy.external as sce
    # import palantir
    # import matplotlib.pyplot as plt

    # expression_data = sc.read_csv('/home/pengrui/STGRN/output/mESC/MTGRN/mtgrn_expression.csv')
    # sc.pp.pca(expression_data)
    
    # dm_res = palantir.utils.run_diffusion_maps(expression_data)
    # ms_data = palantir.utils.determine_multiscale_space(expression_data)

    # imputed_x = palantir.utils.run_magic_imputation(expression_data)

    # start_cell = expression_data.obs.index[0]
    # terminal_states = expression_data.obs.index[-1]

    # pr_res = palantir.core.run_palantir(
    #     expression_data,
    #     early_cell=start_cell,
    #     terminal_states = [terminal_states],
    #     num_waypoints=500,
    # )

    # cell_names = expression_data.obs.index
    # mask = np.ones(len(cell_names), dtype=bool)
    # expression_data.obsm['branch_masks'] = pd.DataFrame({'ALL':mask},index=cell_names)
    
    # genes_common = ['POU5F1']

    # # Palantir uses MAGIC to impute the data for visualization and determining gene expression trends.
    
    # gene_trends = palantir.presults.compute_gene_trends(expression_data,
    #                                                     expression_key='MAGIC_imputed_data')
    
    # palantir.plot.plot_gene_trends(expression_data, genes_common)
    # plt.savefig('/home/pengrui/gene_trend.jpg')
        




    