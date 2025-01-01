import argparse
import os
import multiprocessing
import pandas as pd
import scanpy as sc
import networkx as nx
import copy
import celloracle as co
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from scipy.stats import pearsonr
import numpy as np
import pytorch_warmup as warmup
from GRN期刊.utils import *
import itertools
from Data import Train_data, Test_data
from model import MTGRN
from baseline_models import baseline_compare

seed_everything(2024)
change_into_current_py_path()     
logger = set_logger()      

def preprocess_data(args,logger):
    logger.info(f'\n****************preprocessing data**********************')

    save_path = os.path.join(args.output_dir,args.dataset_name)
    os.makedirs(save_path,exist_ok=True)

    pval_cutoff = 0.01
    select_variable_genes = 2000

    
    expr_df = pd.read_csv(args.data_path, header=0, index_col=0)
    expr_df.index = expr_df.index.map(lambda x: x.upper()) 

    
    different_gene = pd.read_csv(args.gene_order,header=0,index_col=0)
    different_gene.index = different_gene.index.map(lambda x: x.upper())
    
    
    expr_variable_genes = sorted(set(different_gene.index.values) & set(expr_df.index.values))
    extra_genes = sorted(set(different_gene.index.values) - set(expr_df.index.values))
    if len(extra_genes) != 0:
        print("\nWARNING: %d variable genes are not in ExpressionData.csv:" % (len(extra_genes)))
        print(extra_genes)
        different_gene = different_gene.loc[expr_variable_genes]

    
    pval_col = different_gene.columns[0]
    different_gene.sort_values(by=pval_col, ascending=True,inplace=True)

   
    pval_cutoff = pval_cutoff / float(len(different_gene.index))
    print("\nUsing the BF-corrected p-value cutoff of %s (%s / %s genes)" % (
        pval_cutoff, pval_cutoff*float(len(different_gene.index)), len(different_gene.index)))

    
    variable_genes = different_gene[different_gene[pval_col] < pval_cutoff].index.values
    print("\n%d genes pass pval_cutoff of %s" % (len(variable_genes), pval_cutoff))
    
    
    different_gene = different_gene.filter(items = variable_genes, axis='index')

    if select_variable_genes > len(different_gene):    
        end_variable_genes = different_gene.index.tolist()
    else:
        var_col = different_gene.columns[1]
        different_gene.sort_values(by=var_col, inplace=True, ascending = False)
        end_variable_genes = different_gene.iloc[:select_variable_genes].index.tolist()

   
    print("\nRestricting to %d genes" % (len(end_variable_genes)))
    
    
    expr_df = expr_df.loc[end_variable_genes]
    
    
    tf_df = pd.read_csv(args.tf_path, header=0)
    tfs = tf_df[tf_df.columns[0]].drop_duplicates()
    tfs = tfs.map(lambda x: x.upper())
    end_tf = sorted(set(end_variable_genes) & set(tfs))

    
    print('\nloading ground truth...')
    gt_grn = pd.read_csv(args.gt_path)
    
    
    gt_grn = gt_grn[(gt_grn.Gene1.isin(end_tf)) & (gt_grn.Gene2.isin(end_variable_genes))]
    # Remove self-loops.
    gt_grn = gt_grn[gt_grn.Gene1 != gt_grn.Gene2]
    # Remove duplicates (there are some repeated lines in the ground-truth networks!!!). 
    gt_grn.drop_duplicates(keep = 'first', inplace=True)
    
   
    allNodes = sorted(set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique())))
    end_tf = sorted(set(allNodes) & set(end_tf))

    
    expr_df = expr_df[expr_df.index.isin(allNodes)]
    print("\nNew shape of Expression Data %d x %d" % (expr_df.shape[0],expr_df.shape[1]))
    
    
    expression_path = os.path.join(save_path,'expression.csv')
    expr_df.T.to_csv(expression_path)       #cell * gene

    
    gt_grn_path = os.path.join(save_path,'gt_grn.csv')
    gt_grn.to_csv(gt_grn_path,index=False)

    
    pd.DataFrame({'tf':end_tf}).to_csv(os.path.join(args.output_dir,args.dataset_name,'end_tf.csv'))

    
    pd.DataFrame({'allNodes':allNodes}).to_csv(os.path.join(args.output_dir,args.dataset_name,'allNodes.csv'))

    return True

def parse_args():
    parser = argparse.ArgumentParser(description='expriments for MTGRN')

    parser.add_argument('--data_source', type=str, required=True, choices=['BEELINE', 'others'], help='Specify the data source')
    parser.add_argument('--dataset_name',type=str,required=True, help='name of dataset')
    parser.add_argument('--data_path', type=str, required=True, help='expression data file') 
    parser.add_argument('--output_dir', type=str, required=True, help='location to store results')
    parser.add_argument('--tf_path',type=str, required=True, help='tf file')
    parser.add_argument('--species',type=str,required=True,choices=['mouse','human'],help='species')
    
    parser.add_argument('--gt_path',type=str,help='ground truth file') 
    parser.add_argument('--time_info',type=str, help='PseudoTime file')
    parser.add_argument('--gene_order',type=str)
    parser.add_argument('--logFC_path',type=str,help='logFC')

    #测试
    parser.add_argument('--baseline',action='store_true',help='compare with baseline')
    parser.add_argument('--index',type=int,default=-1)
    
    #model
    parser.add_argument('--cuda_index',type=int)
    parser.add_argument('--input_length',type=int, default=16, help='input length')
    parser.add_argument('--predict_length',type=int,default=16,help='output length')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of hidden states of time layer transformer')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
    parser.add_argument('--heads', type=int, default=4, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--batch_size',type=int,default=4,help='batch size')
    parser.add_argument('--encoder_layers',default=4,type=int,help='layers of transformer')
    parser.add_argument('--epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--patience',type=int,default=3)

    args = parser.parse_args()

    return args

def preprocess_data_for_mtgrn(args):
    expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')
    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')

    expr_df = pd.read_csv(expression_path)
    gt_grn = pd.read_csv(gt_grn_path)

    allNodes = pd.read_csv(os.path.join(args.output_dir,args.dataset_name,'allNodes.csv'))['allNodes'].unique()
    end_tf = pd.read_csv(os.path.join(args.output_dir,args.dataset_name,'end_tf.csv'))['tf'].unique()

    logFC_value = pd.read_csv(args.logFC_path)
    logFC_value['primerid'] = logFC_value['primerid'].map(lambda x: x.upper())
    logFC_value = logFC_value.loc[logFC_value['primerid'].isin(allNodes)]

   
    print('\nloading prior knowledge...')
    # prior_grn = pd.read_csv('./%s_prior_GRN.csv' % args.species)
    
    if args.species == 'human':
        prior_grn = pd.read_csv('Prior/network_human.csv',index_col=0)
    else:
        prior_grn = pd.read_csv('Prior/network_mouse.csv',index_col=0)
    
    prior_grn['from'] = prior_grn['from'].map(lambda x: x.upper())
    prior_grn['to'] = prior_grn['to'].map(lambda x: x.upper())

   
    prior_grn = prior_grn.loc[prior_grn['from'].isin(end_tf)
                        & prior_grn['to'].isin(allNodes), :]
    prior_grn = prior_grn.loc[prior_grn['from'] != prior_grn['to']]
    prior_grn = prior_grn.drop_duplicates(subset=['from', 'to'], keep='first')
    prior_grn = prior_grn.iloc[:,:2]
    prior_grn.columns = ['Gene1','Gene2']

    
    gene_id_map = {x:y for y,x in enumerate(allNodes)}   
    id_gene_df = pd.DataFrame({'id': range(len(allNodes)),
                                    'geneName': allNodes},index=range(len(allNodes)))
    
   
    no_regulate_genes = sorted(set(allNodes) - set(prior_grn.Gene2.unique()))
    no_regulate_genes_index = [gene_id_map[x] for x in no_regulate_genes]

    
    prior_grn['Gene1_id'] = prior_grn['Gene1'].map(gene_id_map)
    prior_grn['Gene2_id'] = prior_grn['Gene2'].map(gene_id_map)

    
    prior_mask = np.ones((len(allNodes),len(allNodes)))   
    prior_mask[prior_grn['Gene2_id'],prior_grn['Gene1_id']] = 0   

    print('\nloading time info...')
    time_info = pd.read_csv(args.time_info)
    time_info = time_info.rename(columns={'Unnamed: 0':'cell'})
    time_info = time_info.sort_values(by='PseudoTime',ascending=True) 

    if args.dataset_name == "hHep":
        time_info['true_time'] = time_info['cell'].map(lambda x: x.split('_')[1])
        time_info.reset_index(drop=True,inplace=True)
        time_list = list()
        for time in ['ipsc','de','he1','he2','ih1','mh1','mh2']:
            time_list.append(time_info[time_info['true_time'] == time])
        time_info = pd.concat(time_list,axis=0)
        cell_index_sorted_by_time = time_info['cell']

    elif args.dataset_name == 'hESC':
        time_info['true_time'] = time_info['cell'].map(lambda x: x.split('_')[1])
        time_info.reset_index(drop=True,inplace=True)
        time_list = list()
        for time in ['00hb4s','12h','24h','36h','72h','96h']:
            time_list.append(time_info[time_info['true_time'] == time])
        time_info = pd.concat(time_list,axis=0)
        cell_index_sorted_by_time = time_info['cell']
    
    elif args.dataset_name == "mESC":
        time_info['true_time'] = time_info['cell'].map(lambda x: x.split('_')[2])
        time_info.reset_index(drop=True,inplace=True)
        time_list = list()
        for time in ['00h','12h','24h','48h','72h']:
            time_list.append(time_info[time_info['true_time'] == time])
        time_info = pd.concat(time_list,axis=0)
        cell_index_sorted_by_time = time_info['cell']
    
    elif args.dataset_name == 'mDC':
        time_info['true_time'] = time_info['cell'].map(lambda x: x.split('_')[1])
        time_info.reset_index(drop=True,inplace=True)
        time_list = list()
        for time in ['1h','2h','4h','6h']:
            time_list.append(time_info[time_info['true_time'] == time])
        time_info = pd.concat(time_list,axis=0)
        cell_index_sorted_by_time = time_info['cell']
    else:
        time_info['true_time'] = time_info['cell'].map(lambda x: '_'.join(x.split("_")[:-1]))
        time_info.reset_index(drop=True,inplace=True)
        time_list = list()
        for time in ['LT_HSC','HSPC','Prog']:
            time_list.append(time_info[time_info['true_time'] == time])
        time_info = pd.concat(time_list,axis=0)
        cell_index_sorted_by_time = time_info['cell']

    expr_df = expr_df.set_index('Unnamed: 0').loc[cell_index_sorted_by_time]        
    
    expression_data = expr_df.values # cell * gene
    dropout_mask = (expression_data != 0).astype(int)

    size=[args.input_length,args.predict_length]

    os.makedirs(os.path.join(args.output_dir,args.dataset_name,'MTGRN'),exist_ok=True)
    expr_df.to_csv(os.path.join(args.output_dir,args.dataset_name,'MTGRN','mtgrn_expression.csv'))
    cell_index_sorted_by_time.to_csv(os.path.join(args.output_dir,args.dataset_name,'MTGRN','mtgrn_cell_sort.csv'))

    return expression_data, dropout_mask, size, prior_mask, id_gene_df,no_regulate_genes_index,logFC_value,prior_grn

def val(model, args, val_loader, criterion, prior_mask,no_tf_genes_index):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for (batch_x, batch_y, batch_y_mask) in val_loader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            batch_y_mask = batch_y_mask.to(args.device)

            outputs, _, = model(batch_x, prior_mask,no_tf_genes_index)
            loss = criterion(outputs * batch_y_mask, batch_y * batch_y_mask)
            total_loss.append(loss.item())
    
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

def get_grn(predict_grn,id_gene_df,logFC_value,prior_grn):
    
    edge_df = predict_grn[['Gene1','Gene2']]
    edge_index = edge_df.values
    
    ## Scaled the attention coefficients for global ranking
    g = nx.from_edgelist(edge_index, create_using=nx.DiGraph)
    degree = pd.DataFrame(g.out_degree(), columns=['index', 'degree'])
    
    temp_grn = pd.merge(predict_grn,degree,left_on='Gene1',right_on='index',how='left')
    logFC_value['logFC_gene'] = logFC_value['primerid'].map(lambda x: id_gene_df[id_gene_df['geneName'] == x]['id'].values[0])
    
    weighted_grn = pd.merge(temp_grn,logFC_value,left_on='index',right_on='logFC_gene',how='left')
    weighted_grn = weighted_grn.dropna()
    weighted_grn['target_gene'] = weighted_grn['Gene2'].map(lambda x: id_gene_df[id_gene_df['id'] == x]['geneName'].values[0])    
    
    weighted_grn['target_gene_different'] = 0
    for index,row in weighted_grn.iterrows():
        target_gene = row['target_gene']
        try:
            logFC = logFC_value[logFC_value['primerid'] == target_gene]['logFC'].values[0]
            if logFC >= 1.0:
                weighted_grn.loc[index,'target_gene_different'] = 1
        except:
            pass
    
    temp = weighted_grn.groupby('Gene1')['target_gene_different'].sum().reset_index().rename(columns={'target_gene_different':'whole_target_gene_different'})
    weighted_grn = pd.merge(weighted_grn,temp,on='Gene1',how='left')
    weighted_grn['whole_target_gene_different'] = weighted_grn['whole_target_gene_different'].replace(0,0.5)
    weighted_grn['weights_combined'] = weighted_grn['value'] * weighted_grn['logFC'] * weighted_grn['degree'] * weighted_grn['whole_target_gene_different']

    ## To networkx
    G_nx = nx.from_pandas_edgelist(weighted_grn, source='Gene1', target='Gene2', edge_attr='weights_combined',
                                    create_using=nx.DiGraph)
    
    largest_components = max(nx.weakly_connected_components(G_nx), key=len)
    G_nx = G_nx.subgraph(largest_components)

    ## Use gene name as index
    mappings = id_gene_df.loc[id_gene_df['id'].isin(G_nx.nodes()), ['id', 'geneName']]
    mappings = {idx: geneName for (idx, geneName) in np.array(mappings)}
    G_nx = nx.relabel_nodes(G_nx, mappings)
    
    ## Convert G_nx to a DataFrame
    end_grn = nx.to_pandas_edgelist(G_nx,source='Gene1',target='Gene2')
    
    return end_grn

def test(args,save_path,checkpoint_path,expression_data,id_gene_df,size,prior_mask,no_tf_genes_index,logFC_value,prior_grn):
    print("\ntesting...")
    test_data_set = Test_data(expression_data,size)
    test_data_loader = DataLoader(
        test_data_set,
        batch_size= 1,
        shuffle=False,
        num_workers=args.num_workers)
    
    model = MTGRN(args.input_length,
                    args.predict_length,
                    args.d_model,
                    args.d_ff,
                    args.heads,
                    args.dropout,
                    args.encoder_layers).to(args.device)

    model.load_state_dict(torch.load(checkpoint_path))

    model.eval()
    with torch.no_grad():
        attention_list = list()
        for (batch_x) in tqdm(test_data_loader):
            batch_x = batch_x.float().to(args.device)
            outputs, attention = model(batch_x, prior_mask,no_tf_genes_index)
            attention_list.append(attention.squeeze())

        attention_value = torch.stack(attention_list).cpu().numpy()
        mean_attention = np.mean(attention_value, axis=0)
        mean_attention = np.mean(mean_attention, axis=0)

        num_nodes = mean_attention.shape[0]
        mat_indicator_all = np.zeros([num_nodes, num_nodes])
        
        mat_indicator_all[abs(mean_attention) > 0] = 1
        idx_rec, idx_send = np.where(mat_indicator_all)
        predict_grn = pd.DataFrame(
            {'Gene1': idx_send, 'Gene2': idx_rec, 'value': mean_attention[idx_rec,idx_send]})

        predict_grn = predict_grn[predict_grn['Gene1'] != predict_grn['Gene2']]
        predict_grn = get_grn(predict_grn,id_gene_df,logFC_value,prior_grn)

        predict_grn = predict_grn.sort_values('weights_combined',ascending=False)
        predict_grn.to_csv(os.path.join(save_path,'grn.csv'))

        auc,aupr = computeScores(os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv'),os.path.join(save_path,'grn.csv'))
        
        if args.index != -1:
            result = f'input_length:{args.input_length},predict_length:{args.predict_length},d_model:{args.d_model},heads:{args.heads},dropout:{args.dropout},layers:{args.encoder_layers},auc:{auc},aupr:{aupr}'
            with open(os.path.join(args.output_dir,args.dataset_name,'MTGRN',f'results_{args.index}.txt'), 'a') as file:
                file.write(result + '\n')
        else:
            print(auc,aupr)
        
        return True

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.cuda_index}')
    args.device = device
    
    preprocess_data(args,logger)
    
    if args.baseline:
        baseline_compare(args,logger)
    else:
        # preprocess MTGRN data
        expression_data, dropout_mask, size, prior_mask, id_gene_df,no_tf_genes_index,logFC_value,prior_grn = preprocess_data_for_mtgrn(args) # expression_data (cell * gene)

        train_data_set = Train_data(expression_data,dropout_mask,size,'train')
        train_data_loader = DataLoader(
            train_data_set,
            batch_size= args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)
        
        val_data_set = Train_data(expression_data,dropout_mask,size,'val')
        val_data_loader = DataLoader(
            val_data_set,
            batch_size= int(args.batch_size / 4),
            shuffle=False,
            num_workers=args.num_workers)

        model = MTGRN(args.input_length,
                args.predict_length,
                args.d_model,
                args.d_ff,
                args.heads,
                args.dropout,
                args.encoder_layers).to(args.device)
        
        save_path = os.path.join(args.output_dir,args.dataset_name,'MTGRN')
        os.makedirs(save_path, exist_ok=True)

        model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.epochs)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        prior_mask = torch.from_numpy(prior_mask).to(args.device,dtype=torch.bool)

        for _ in tqdm(range(args.epochs)):
            model.train()
            train_loss_list = list()
            for i, (batch_x, batch_y, batch_y_mask) in enumerate(train_data_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(args.device)
                batch_y = batch_y.float().to(args.device)
                batch_y_mask = batch_y_mask.to(args.device)

                outputs, _, = model(batch_x,prior_mask,no_tf_genes_index)

                loss = criterion(outputs * batch_y_mask, batch_y * batch_y_mask)

                train_loss_list.append(loss.item())
                loss.backward()
                model_optim.step()
                                
            lr_scheduler.step()            
            val_loss = val(model, args, val_data_loader, criterion, prior_mask,no_tf_genes_index)

            if args.index != -1:
                checkpoint_path = os.path.join(save_path,f'checkpoint_{args.index}.pth')
            else:
                checkpoint_path = os.path.join(save_path,f'checkpoint.pth')
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
        torch.save(early_stopping.best_model,checkpoint_path)
        test(args,save_path,checkpoint_path,expression_data,id_gene_df,size,prior_mask,no_tf_genes_index,logFC_value,prior_grn)

if __name__ == "__main__":
    main()    