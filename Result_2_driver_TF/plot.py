import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

current_path = os.path.dirname(__file__)
os.chdir(current_path)

sns.set_theme(font='Times New Roman',font_scale=1.4)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def change_into_current_py_path():
    current_file_path = __file__
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

def master_regulator(regulon_score,cell_sort_df):
    import pandas as pd
    import statsmodels.stats.multitest as smm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from scipy import stats
    
    # label=pd.DataFrame(cell_sort_df['time_label'].values.tolist(),columns=['time_label'])
    # label = label.astype(str)
    
    # label_set=np.array(list(set(label['time_label'].values)))
    # t_test_results = np.zeros((regulon_score.shape[0],3*len(label_set)))
    # for j in range(len(label_set)):
    #     idx=regulon_score.columns[np.isin(label['time_label'],label_set[j])]
    #     X=regulon_score[idx]
    #     idx=regulon_score.columns[np.isin(label['time_label'],label_set[j])==0]
    #     Y=regulon_score[idx]
    #     # Initialize an empty DataFrame to store the t-test results
    #     # Iterate over the columns/variables in X and Y
    #     for i in range(X.shape[0]):
    #         row= regulon_score.index[i]
    #     # Perform the t-test between X and Y for the current variable
    #         t_stat, p_value = stats.ttest_ind(X.loc[row], Y.loc[row],alternative='greater')   
    #     # Append the results to the DataFrame
    #         t_test_results[i,3*j+1] = p_value 
    #         t_test_results[i,3*j] = t_stat
    #     t_test_results[np.isnan(t_test_results)]=1
    #     t_test_results[:,3*j+2]=smm.multipletests(t_test_results[:,3*j+1], method='fdr_bh')[1]
    # t_test_results=pd.DataFrame(t_test_results,index=regulon_score.index)
    
    # col=[0 for kk in range(len(label_set)*3)]
    # for j in range(len(label_set)):
    #     col[3*j]=label_set[j]+'_t_stat'
    #     col[3*j+1]=label_set[j]+'_p_value'
    #     col[3*j+2]=label_set[j]+'_adj_p'
    # t_test_results.columns=col
    # return t_test_results

    t_test_results = np.zeros((regulon_score.shape[0],1))
    
    for j in range(regulon_score.shape[0]):
        row = regulon_score.index[j]
        activity_score = pd.DataFrame(regulon_score.loc[row]).reset_index()
        time_score = pd.merge(activity_score,cell_sort_df,left_on='index',right_on='cell',how='inner')
        model_gene=ols(f'{row}~time_label',time_score).fit() # ols（）创建一线性回归分析模型
        anova_table=anova_lm(model_gene)  # anova_lm（）函数创建模型生成方差分析表
        t_test_results[j, 0] = anova_table['PR(>F)'][0]
        
    t_test_results_df = pd.DataFrame(t_test_results, index=regulon_score.index)
    t_test_results_df = t_test_results_df.dropna(axis=0)
    t_test_results_df.columns=['p_value']
    t_test_results_df['adj_p'] = smm.multipletests(t_test_results_df['p_value'].values, method='fdr_bh')[1]
    t_test_results_df=t_test_results_df.sort_values(by='adj_p',ascending=True)
    return t_test_results_df

def plot_RGM_activity_heatmap(regulon_score,time_label,save_path):
    """
    Plot clustermap of RGM activity matrix.
    If `cell_label` is provided, cells are ordered by cell clusters, else cells are ordered by hierarchical clustering.
    """

    auc_mtx = regulon_score
    auc_mtx.index = [gene.capitalize() for gene in auc_mtx.index]

    # Create a categorical palette to identify the networks
    network_lable = time_label
    network_pal = sns.husl_palette(len(set(network_lable)), h=.5)
    network_lut = dict(zip(map(str, set(network_lable)), network_pal))
    network_colors = pd.Series(list(network_lable), index=auc_mtx.columns).astype(str).map(network_lut)
    
    # plot clustermap (n_cell * n_gene)
    f = plt.figure()
    sns.set_theme(font='Times New Roman',font_scale=1.4)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    # sns.set_theme(font_scale=f.get_dpi()/150)
    g = sns.clustermap(auc_mtx, method='ward', z_score=0, vmin=-1.5, vmax=1.5, linecolor='black', cmap='RdBu_r',
                        col_cluster=False, row_cluster=True, col_colors=network_colors,
                        figsize=(10,12),        #4.5, 0.15 * auc_mtx.shape[1]
                        xticklabels=False, yticklabels=True, dendrogram_ratio=0.12,
                        cbar_pos=(0.15, 0.92, 0.2, 0.02),
                        cbar_kws={'orientation': 'horizontal'})

    # g.cax.set_visible(True)
    # g.ax_heatmap.set_ylabel('Regulon-like gene modules')
    # g.ax_heatmap.set_xlabel('Cells')
    # g.ax_heatmap.yaxis.set_minor_locator(ticker.NullLocator())

    # for label in map(str, set(network_lable)):
    #     g.ax_col_dendrogram.bar(0, 0, color=network_lut[label], label=label, linewidth=0)
    # g.ax_col_dendrogram.legend(title='Cell types', loc="upper left", ncol=1,
    #                             bbox_to_anchor=(0.80, 1.10), facecolor='white')
    # g.ax_row_dendrogram.set_visible(False)
    plt.savefig(save_path)
    
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
        each_alg = 'MTGRN'

        auc_mtx_out = pd.read_csv('aucell.csv',sep=',')
        auc_mtx_out = auc_mtx_out.set_index('Rowname')
        
        #读取细胞顺序
        cell_sort_path = os.path.join('input','mESC','mtgrn_cell_sort.csv')
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
        
        save_path = './RGM_results'
        os.makedirs(save_path,exist_ok=True)

        plot_RGM_activity_heatmap(auc_mtx_out,time_label,os.path.join(save_path,f'{each_dataset}_RGM.svg'))
        result.to_csv(os.path.join(save_path,f't_test_driver_gene.csv'))

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
        




    