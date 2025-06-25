import pandas as pd
import os 
from utils import *
import celloracle as co

import sys
sys.path.append('Baseline_models/inferelator')
from inferelator import inferelator_workflow

seed_everything(2024)

def run_CEFCON(args,algorithm,logger):
    logger.info(f'\n****************running CEFCON**********************')
    
    expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')
    output_path = os.path.join(args.output_dir,args.dataset_name,algorithm)
    os.makedirs(output_path,exist_ok=True)

    if args.species == 'human':
        prior_grn_path = 'Prior/network_human_merged.csv'
    else:
        prior_grn_path = 'Prior/network_mouse_merged.csv'

    os.system(f'cefcon --input_expData {expression_path} --input_priorNet {prior_grn_path} --out_dir {output_path} --seed 2024 --species {args.species} --cuda 5 --repeats 1')
    
    return True

def run_GRNBoost2(args,algorithm,logger):
    logger.info(f'\n****************running GRNBoost2**********************')

    save_path = os.path.join(args.output_dir,args.dataset_name,algorithm)
    os.makedirs(save_path,exist_ok=True)
    
    gene_expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')

    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)
    
    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))

    
    tf_df = pd.read_csv(args.tf_path, header=0)
    tfs = tf_df[tf_df.columns[0]].drop_duplicates()
    tfs = tfs.map(lambda x: x.upper())
    end_tf = allNodes & set(tfs)

    tf_name_path = os.path.join(save_path,'tf_names.txt')
    with open(tf_name_path,'w') as f:
        for item in end_tf:
            f.write("%s\n" % item) 
    
    save_grn_path = os.path.join(save_path,'grn.csv')

    os.system(f'python Baseline_models/GRNBoost2/arboreto_with_multiprocessing.py \
                {gene_expression_path} \
                {tf_name_path} \
                --method grnboost2 \
                --output {save_grn_path} \
                --num_workers 20 \
                --seed 2024')

    return True

def run_NetREX(args,algorithm,logger):
    logger.info(f'\n****************running NetREX**********************')

    save_path = os.path.join(args.output_dir,args.dataset_name,algorithm)
    os.makedirs(save_path,exist_ok=True)

    expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')
    expr_df = pd.read_csv(expression_path).T

    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)
    
    print('\nloading prior knowledge...')
    
    if args.species == 'human':
        prior_grn = pd.read_csv('Prior/network_human_merged.csv',index_col=None)
    else:
        prior_grn = pd.read_csv('Prior/network_mouse_merged.csv',index_col=None)

    prior_grn.drop(columns=['Unnamed: 0'],inplace=True)
    prior_grn.reset_index(drop=True,inplace=True)
    
    prior_grn['from'] = prior_grn['from'].map(lambda x: x.upper())
    prior_grn['to'] = prior_grn['to'].map(lambda x: x.upper())

    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))

    
    tf_df = pd.read_csv(args.tf_path, header=0)
    tfs = tf_df[tf_df.columns[0]].drop_duplicates()
    tfs = tfs.map(lambda x: x.upper())
    end_tf = allNodes & set(tfs)

    
    prior_grn = prior_grn.loc[prior_grn['from'].isin(end_tf)
                        & prior_grn['to'].isin(allNodes), :]
    prior_grn = prior_grn.drop_duplicates(subset=['from', 'to'], keep='first')
    prior_grn = prior_grn.iloc[:,:2]
    prior_grn.columns = ['Gene1','Gene2']
    prior_grn['value'] = 1
    prior_grn = prior_grn.pivot(index='Gene2', columns='Gene1', values='value').fillna(0)

    expr_df = expr_df.loc[prior_grn.index]

    
    exp_data_path = os.path.join(save_path, "expression.txt")
    prior_path = os.path.join(save_path, "prior_grn.txt")

    prior_grn.to_csv(prior_path, sep='\t', index=True, header=True)
    expr_df.to_csv(exp_data_path, sep='\t', header=False, index=True)

    os.system(f"python Baseline_models/NetREX/NetREX.py -e {exp_data_path} -p {prior_path} -o {save_path} -k 0.6")
    
    return True

def run_Random(args,algorithm,logger):
    logger.info(f'\n****************running Random**********************')
    os.makedirs(os.path.join(args.output_dir,args.dataset_name,algorithm),exist_ok=True)
    
    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)

    
    tf_path = os.path.join(args.output_dir,args.dataset_name,'end_tf.csv')
    tf_df = pd.read_csv(tf_path)
    tf_list = tf_df['tf'].unique()
    
    
    possibleEdges = list(permutations(np.unique(gt_grn.loc[:, ['Gene1', 'Gene2']]),
                                        r=2))
    TrueEdgeDict = pd.DataFrame(possibleEdges)
    TrueEdgeDict.columns = ['Gene1','Gene2']
    TrueEdgeDict = TrueEdgeDict[TrueEdgeDict['Gene1'].isin(tf_list)]
    TrueEdgeDict = TrueEdgeDict.drop_duplicates(subset=['Gene1', 'Gene2'], keep='first')
    TrueEdgeDict = TrueEdgeDict.iloc[:,:2]

    random_numbers = np.random.rand(len(TrueEdgeDict))
    TrueEdgeDict['weights_combined'] = random_numbers

    
    TrueEdgeDict.to_csv(os.path.join(args.output_dir,args.dataset_name,algorithm,'grn.csv'),index=False)

def run_prior_Random(args,algorithm,logger):
    logger.info(f'\n****************running prior Random**********************')
    os.makedirs(os.path.join(args.output_dir,args.dataset_name,algorithm),exist_ok=True)
    
    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)
    
    
    print('\nloading prior knowledge...')
    
    if args.species == 'human':
        prior_grn = pd.read_csv('Prior/network_human_merged.csv')
    else:
        prior_grn = pd.read_csv('Prior/network_mouse_merged.csv')
    
    prior_grn['from'] = prior_grn['from'].map(lambda x: x.upper())
    prior_grn['to'] = prior_grn['to'].map(lambda x: x.upper())

    
    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))

    
    prior_grn = prior_grn.loc[prior_grn['from'].isin(allNodes)
                        & prior_grn['to'].isin(allNodes), :]
    prior_grn = prior_grn.drop_duplicates(subset=['from', 'to'], keep='first')
    prior_grn = prior_grn.iloc[:,1:3]
    prior_grn.columns = ['Gene1','Gene2']

    random_numbers = np.random.rand(len(prior_grn))
    prior_grn['weights_combined'] = random_numbers

    
    prior_grn.to_csv(os.path.join(args.output_dir,args.dataset_name,algorithm,'grn.csv'),index=False)

    return True

def run_GENIE3(args,algorithm,logger):
    logger.info(f'\n****************running GENIE3**********************')

    save_path = os.path.join(args.output_dir,args.dataset_name,algorithm)
    os.makedirs(save_path,exist_ok=True)
    
    gene_expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')

    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)
    
    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))

    
    tf_df = pd.read_csv(args.tf_path, header=0)
    tfs = tf_df[tf_df.columns[0]].drop_duplicates()
    tfs = tfs.map(lambda x: x.upper())
    end_tf = allNodes & set(tfs)

    tf_name_path = os.path.join(save_path,'tf_names.txt')
    with open(tf_name_path,'w') as f:
        for item in end_tf:
            f.write("%s\n" % item) 
    
    save_grn_path = os.path.join(save_path,'grn.csv')

    os.system(f'/home/pengrui/mambaforge/envs/grn/bin/python ./Baseline_models/GRNBoost2/arboreto_with_multiprocessing.py \
                {gene_expression_path} \
                {tf_name_path} \
                --method genie3 \
                --output {save_grn_path} \
                --num_workers 20 \
                --seed 2024')

    return True

def run_Inferelator(args, algorithm, logger):
    logger.info(f'\n****************running Inferelator**********************')
    
    save_path = os.path.join(args.output_dir, args.dataset_name, algorithm)
    os.makedirs(save_path, exist_ok=True)
    
    expression_path = os.path.join(args.output_dir, args.dataset_name, 'expression.csv')
    
    gt_grn_path = os.path.join(args.output_dir, args.dataset_name, 'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)
    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))
    
    tf_df = pd.read_csv(args.tf_path, header=0)
    tfs = tf_df[tf_df.columns[0]].drop_duplicates()
    tfs = tfs.map(lambda x: x.upper())
    end_tf = allNodes & set(tfs)
    
    print('\nloading prior knowledge...')
    if args.species == 'human':
        prior_grn = pd.read_csv('Prior/network_human_merged.csv', index_col=None)
    else:
        prior_grn = pd.read_csv('Prior/network_mouse_merged.csv', index_col=None)
    
    prior_grn.drop(columns=['Unnamed: 0'], inplace=True)
    prior_grn.reset_index(drop=True, inplace=True)
    
    prior_grn['from'] = prior_grn['from'].map(lambda x: x.upper())
    prior_grn['to'] = prior_grn['to'].map(lambda x: x.upper())
    
    logger.info(f"Prior GRN before filtering: {len(prior_grn)} edges")
    logger.info(f"Available TFs: {len(end_tf)}, Available genes: {len(allNodes)}")
    logger.info(f"Sample TFs: {list(end_tf)[:5]}")
    logger.info(f"Sample genes: {list(allNodes)[:5]}")
    logger.info(f"Sample prior edges before filtering:\n{prior_grn.head()}")
    
    prior_grn = prior_grn.loc[prior_grn['from'].isin(end_tf) & prior_grn['to'].isin(allNodes), :]
    logger.info(f"Prior GRN after filtering: {len(prior_grn)} edges")
    
    prior_grn = prior_grn.drop_duplicates(subset=['from', 'to'], keep='first')
    prior_grn = prior_grn[['from', 'to']]
    prior_grn.columns = ['Gene1', 'Gene2']
    
    logger.info(f"Final prior GRN shape: {prior_grn.shape}")
    logger.info(f"Sample final prior edges:\n{prior_grn.head()}")
    
    expr_df = pd.read_csv(expression_path, index_col=0)
    expr_df_transposed = expr_df.T  
    expr_tsv_path = os.path.join(save_path, 'expression.tsv')
    expr_df_transposed.to_csv(expr_tsv_path, sep='\t')
    
    logger.info(f"Original expression data shape (cells x genes): {expr_df.shape}")
    logger.info(f"Transposed expression data shape (genes x cells): {expr_df_transposed.shape}")

    tf_names_path = os.path.join(save_path, 'tf_names.tsv')
    tf_names_df = pd.DataFrame(list(end_tf), columns=['tf'])
    tf_names_df.to_csv(tf_names_path, sep='\t', index=False, header=False)
    
    all_genes = list(allNodes)
    tf_list = list(end_tf)
    
    import numpy as np
    prior_matrix = pd.DataFrame(
        data=np.zeros((len(all_genes), len(tf_list))), 
        index=all_genes,
        columns=tf_list
    )
    
    edges_set = 0
    for _, row in prior_grn.iterrows():
        tf_name = row['Gene1']  
        target_gene = row['Gene2']  
        if tf_name in tf_list and target_gene in all_genes:
            prior_matrix.loc[target_gene, tf_name] = 1
            edges_set += 1
    
    logger.info(f"Successfully set {edges_set} edges in prior matrix out of {len(prior_grn)} total edges")
    
    priors_path = os.path.join(save_path, 'priors.tsv')
    prior_matrix.to_csv(priors_path, sep='\t')
    
    logger.info(f"Prior matrix shape: {prior_matrix.shape}")
    logger.info(f"Prior matrix non-zero entries: {(prior_matrix != 0).sum().sum()}")
    
    
    meta_data_path = os.path.join(save_path, 'meta_data.tsv')
    meta_data = pd.DataFrame({
        'sample': expr_df.columns, 
        'condition': ['condition_1'] * len(expr_df.columns)
    })
    meta_data.to_csv(meta_data_path, sep='\t', index=False)
    
    try:
        worker = inferelator_workflow(
            workflow="single-cell",
            regression="bbsr"
        )
        
        worker.set_file_paths(
            input_dir=save_path,
            output_dir=save_path,
            expression_matrix_file='expression.tsv',
            tf_names_file='tf_names.tsv',
            priors_file='priors.tsv',
            meta_data_file='meta_data.tsv'
        )
        
        worker.set_file_properties(expression_matrix_columns_are_genes=False)
        
        worker.set_network_data_flags(use_no_gold_standard=True)
        
        worker.set_run_parameters(num_bootstraps=5, random_seed=2024)
        
        final_network = worker.run()
        
        logger.info(f"Inferelator completed. Result type: {type(final_network)}")
        
        if final_network is not None:
            if hasattr(final_network, 'network'):
                network_df = final_network.network
                logger.info(f"Network DataFrame shape: {network_df.shape}")
                logger.info(f"Network DataFrame columns: {network_df.columns.tolist()}")
                logger.info(f"First few rows:\n{network_df.head()}")
                
                output_network = network_df.copy()
                if 'regulator' in output_network.columns and 'target' in output_network.columns:
                    output_network = output_network.rename(columns={
                        'regulator': 'Gene1', 
                        'target': 'Gene2'
                    })
                
                weight_cols = [col for col in output_network.columns if 'coef' in col.lower() or 'weight' in col.lower()]
                if len(weight_cols) > 0:
                    output_network = output_network.rename(columns={weight_cols[0]: 'weights_combined'})
                elif 'combined_confidences' in output_network.columns:
                    output_network = output_network.rename(columns={'combined_confidences': 'weights_combined'})
                
                if 'weights_combined' not in output_network.columns:
                    output_network['weights_combined'] = 1.0
                
                output_columns = ['Gene1', 'Gene2', 'weights_combined']
                available_columns = [col for col in output_columns if col in output_network.columns.tolist()]
                if len(available_columns) >= 2:  
                    output_network = output_network[available_columns]
                    if 'weights_combined' not in output_network.columns:
                        output_network['weights_combined'] = 1.0
                else:
                    logger.warning(f"No expected columns found. Available columns: {output_network.columns.tolist()}")
                    if len(output_network.columns) >= 2:
                        output_network = output_network.iloc[:, :2].copy()
                        output_network.columns = ['Gene1', 'Gene2']
                        output_network['weights_combined'] = 1.0
                    else:
                        output_network = pd.DataFrame(columns=['Gene1', 'Gene2', 'weights_combined'])
                
                grn_output_path = os.path.join(save_path, 'grn.csv')
                output_network.to_csv(grn_output_path, index=False)
                
                logger.info(f"Inferelator completed successfully. Output saved to {grn_output_path}")
            else:
                logger.warning("InferelatorResults object does not have 'network' attribute")
                empty_df = pd.DataFrame(columns=['Gene1', 'Gene2', 'weights_combined'])
                empty_df.to_csv(os.path.join(save_path, 'grn.csv'), index=False)
        else:
            logger.warning("Inferelator returned empty results")
            empty_df = pd.DataFrame(columns=['Gene1', 'Gene2', 'weights_combined'])
            empty_df.to_csv(os.path.join(save_path, 'grn.csv'), index=False)
    except Exception as e:
        logger.error(f"Error running Inferelator: {str(e)}")
        empty_df = pd.DataFrame(columns=['Gene1', 'Gene2', 'weights_combined'])
        empty_df.to_csv(os.path.join(save_path, 'grn.csv'), index=False)
        return False
    
    return True

def baseline_compare(args,logger):
    run_NetREX(args,'NetREX',logger)
    run_Inferelator(args,'Inferelator',logger)
    run_Random(args,'Random',logger)
    run_prior_Random(args,'Prior_Random',logger)
    run_GENIE3(args,'GENIE3',logger)
    run_GRNBoost2(args,'GRNBoost2',logger)
    run_CEFCON(args,'CEFCON',logger)