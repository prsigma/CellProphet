from utils import *
from tqdm import tqdm
from sklearn.metrics import f1_score

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

change_into_current_py_path() 

datasets = ["mESC"]

TF_list = ['NANOG','SOX2','POU5F1']

root_path = './output'

algorithm = ['MTGRN','Celloracle','NetREX','GRNBoost2','GENIE3','Random']

metric_list = list()
for each_dataset in datasets:
    pred_grn_list = list()
    algorithm_list = list()
    gt_grn_path = f'{root_path}/{each_dataset}/gt_grn.csv'

    for each_algorithm in tqdm(algorithm):
        path = f'{root_path}/{each_dataset}/{each_algorithm}/'
        algorithm_list.append(each_algorithm)
        
        if each_algorithm == 'CEFCON':
            pred_grn_list.append(os.path.join(path,'cell_lineage_GRN.csv'))
            
        else:
            pred_grn_list.append(os.path.join(path,'grn.csv'))

    for TF in TF_list:
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
            pred_grn = pred_grn.reset_index()
            pred_grn = pred_grn[pred_grn['index'].str.split("|").str[0] == TF]

            possibleEdges = list(permutations(np.unique(gt_grn.loc[:, ['Gene1', 'Gene2']]),
                                            r=2))
            
            TrueEdgeDict = pd.DataFrame({'|'.join(p): 0 for p in possibleEdges}, index=['label']).T
            TrueEdgeDict.loc[np.array(gt_grn['Gene1'] + "|" + gt_grn['Gene2']), 'label'] = 1
            TrueEdgeDict = TrueEdgeDict.reset_index()
            TrueEdgeDict = TrueEdgeDict[TrueEdgeDict['index'].str.split("|").str[0] == TF]

            result = pd.merge(pred_grn,TrueEdgeDict,on='index',how='inner')
            result['predict'] = 1

            fpr, tpr, _ = roc_curve(y_true=result['label'], y_score=result['weights_combined'],pos_label=1)
            
            # Compute AUC
            auc = round(roc_auc_score(result['label'], result['weights_combined']),3)
            auc_list.append({'Algorithm':algorithm_list[i],'AUROC':auc})

            # Calculate the average precision score AUPR
            aupr = round(average_precision_score(result['label'], result['weights_combined']),3)
            auprc_list.append({'AUPRC':aupr})

            # Calculate F1 score
            f1 = round(f1_score(y_true=result['label'], y_pred=result['predict']),3)
            f1_list.append({'F1':f1})

            # Calculate precision and recall values
            precision, recall, _ = precision_recall_curve(y_true=result['label'], probas_pred=result['weights_combined'], pos_label=1)
            
            # Plot precision-recall curve
            ax_auprc.plot(recall, precision, color=colors[i], label=algorithm_list[i])
            
            # Plot the ROC curve
            ax_roc.plot(fpr, tpr, color=colors[i], label=algorithm_list[i])

        ax_auprc.set_ylim([0.0, 1.0])
        ax_auprc.set_xlabel('Recall')
        ax_auprc.set_ylabel('Precision')
        ax_auprc.set_title('Precision-Recall Curve')
        ax_auprc.legend(loc='upper right')
        figure_auprc.savefig(os.path.join(f'{root_path}/{each_dataset}/',f"{TF}_pr_curve.png"), format='png', bbox_inches='tight')
        plt.close(figure_auprc)

        ax_roc.plot([0, 1], [0, 1], color='black', linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        figure_roc.savefig(os.path.join(f'{root_path}/{each_dataset}/',f"{TF}_roc_curve.png"), format='png', bbox_inches='tight')
        plt.close(figure_roc)
            
        auc_df = pd.DataFrame(auc_list)
        auprc_df = pd.DataFrame(auprc_list)
        f1_df = pd.DataFrame(f1_list)
        metric_df = pd.concat([auc_df,auprc_df],axis=1)
        metric_df = pd.concat([metric_df,f1_df],axis=1)
        metric_df['TF'] = TF
        metric_list.append(metric_df)

    whole_metric_df = pd.concat(metric_list,axis=0)
    whole_metric_df.to_csv(f'{root_path}/whole_metric_TF.csv',index=False)
        