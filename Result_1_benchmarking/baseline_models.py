import pandas as pd
import os 
from GRN期刊.utils import *
import celloracle as co
seed_everything(2024)

def run_CEFCON(args,algorithm,logger):
    logger.info(f'\n****************running CEFCON**********************')
    
    expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')
    output_path = os.path.join(args.output_dir,args.dataset_name,algorithm)
    os.makedirs(output_path,exist_ok=True)

    if args.species == 'human':
        prior_grn_path = '/home/pengrui/CEFCON/prior_data/network_human_baseline.csv'
    else:
        prior_grn_path = '/home/pengrui/CEFCON/prior_data/network_mouse_baseline.csv'

    os.system(f'/home/pengrui/mambaforge/envs/grn/bin/cefcon --input_expData {expression_path} --input_priorNet {prior_grn_path} --out_dir {output_path} --seed 2024 --species {args.species} --cuda 1 --repeats 1')
    
    return True

def run_Celloracle(args,algorithm,logger):
    logger.info(f'\n****************running Celloracle**********************')

    os.makedirs(os.path.join(args.output_dir,args.dataset_name,algorithm),exist_ok=True)
    
    expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')

    adata = sc.read_csv(expression_path)

    # Normalize gene expression matrix with total UMI count per cell
    sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all')
    # keep raw cont data before log transformation
    adata.raw = adata
    adata.layers["raw_count"] = adata.raw.X.copy()
    # Log transformation and scaling
    sc.pp.scale(adata)
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')

    # Diffusion map
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)

    sc.tl.diffmap(adata)
    # Calculate neihbors again based on diffusionmap 
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')
    # Cell clustering
    sc.tl.louvain(adata, resolution=0.8)
    sc.tl.paga(adata, groups='louvain')
    sc.pl.paga(adata)
    
    sc.tl.draw_graph(adata, init_pos='paga', random_state=123)
    # suppose all cells belong to the same type
    adata.obs["cell_type"] = ['all' for _ in adata.obs.louvain]
    adata.obs["louvain_annot"] = ['all' for i in adata.obs.louvain]
    
    if args.species == 'human':
        base_GRN = co.data.load_human_promoter_base_GRN()
        base_GRN["gene_short_name"] = base_GRN["gene_short_name"].apply(lambda x: x.upper())
    else:
        base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()
        base_GRN.columns = list(base_GRN.columns[:2]) + list(base_GRN.columns[2:].str.upper())
        base_GRN["gene_short_name"] = base_GRN["gene_short_name"].apply(lambda x: x.upper())
    
    oracle = co.Oracle()
    # use the unscaled mRNA count for the nput of Oracle object.
    adata.X = adata.layers["raw_count"].copy()

    # Instantiate Oracle object.
    oracle.import_anndata_as_raw_count(adata=adata,
                                    cluster_column_name="louvain_annot",
                                    embedding_name="X_draw_graph_fa")
    
    oracle.import_TF_data(TF_info_matrix=base_GRN)
    # Perform PCA
    oracle.perform_PCA()

    oracle.knn_imputation(n_pca_dims=50, k=1, balanced=True, b_sight=8,
                      b_maxl=4, n_jobs=4)
    # Calculate GRN for each population in "louvain_annot" clustering unit.
    links = oracle.get_links(cluster_name_for_GRN_unit="louvain_annot", alpha=10,
                         verbose_level=10)
    
    links.links_dict['all'][['source', 'target', 'coef_abs']].to_csv(
        os.path.join(args.output_dir,args.dataset_name,algorithm,'grn.csv')
        )
    
    return True

def run_GRNBoost2(args,algorithm,logger):
    logger.info(f'\n****************running GRNBoost2**********************')

    save_path = os.path.join(args.output_dir,args.dataset_name,algorithm)
    os.makedirs(save_path,exist_ok=True)
    
    gene_expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')

    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)
    # ground truth中涉及的基因个数
    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))

    #加载TF列表
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
    
    # 加载先验网络信息
    print('\nloading prior knowledge...')
    
    if args.species == 'human':
        prior_grn = pd.read_csv('/home/pengrui/CEFCON/prior_data/network_human_baseline.csv')
    else:
        prior_grn = pd.read_csv('/home/pengrui/CEFCON/prior_data/network_mouse_baseline.csv')
    
    prior_grn['from'] = prior_grn['from'].map(lambda x: x.upper())
    prior_grn['to'] = prior_grn['to'].map(lambda x: x.upper())

    # ground truth中涉及的基因个数
    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))

    #加载TF列表
    tf_df = pd.read_csv(args.tf_path, header=0)
    tfs = tf_df[tf_df.columns[0]].drop_duplicates()
    tfs = tfs.map(lambda x: x.upper())
    end_tf = allNodes & set(tfs)

    #保证先验网络中的from都是TF,to都是基因表达矩阵中的基因
    prior_grn = prior_grn.loc[prior_grn['from'].isin(end_tf)
                        & prior_grn['to'].isin(allNodes), :]
    prior_grn = prior_grn.drop_duplicates(subset=['from', 'to'], keep='first')
    prior_grn = prior_grn.iloc[:,:2]
    prior_grn.columns = ['Gene1','Gene2']
    prior_grn['value'] = 1
    prior_grn = prior_grn.pivot(index='Gene2', columns='Gene1', values='value').fillna(0)

    expr_df = expr_df.loc[prior_grn.index]

    #保存基因表达矩阵和先验网络
    exp_data_path = os.path.join(save_path, "expression.txt")
    prior_path = os.path.join(save_path, "prior_grn.txt")

    prior_grn.to_csv(prior_path, sep='\t', index=True, header=True)
    expr_df.to_csv(exp_data_path, sep='\t', header=False, index=True)

    os.system(f"/home/pengrui/mambaforge/envs/grn/bin/python ./Baseline_models/NetREX/NetREX.py -e {exp_data_path} -p {prior_path} -o {save_path} -k 0.6")
    
    return True

def run_Random(args,algorithm,logger):
    logger.info(f'\n****************running Random**********************')
    os.makedirs(os.path.join(args.output_dir,args.dataset_name,algorithm),exist_ok=True)
    
    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)

    # 加载tf信息
    tf_path = os.path.join(args.output_dir,args.dataset_name,'end_tf.csv')
    tf_df = pd.read_csv(tf_path)
    tf_list = tf_df['tf'].unique()
    
    #构建边
    possibleEdges = list(permutations(np.unique(gt_grn.loc[:, ['Gene1', 'Gene2']]),
                                        r=2))
    TrueEdgeDict = pd.DataFrame(possibleEdges)
    TrueEdgeDict.columns = ['Gene1','Gene2']
    TrueEdgeDict = TrueEdgeDict[TrueEdgeDict['Gene1'].isin(tf_list)]
    TrueEdgeDict = TrueEdgeDict.drop_duplicates(subset=['Gene1', 'Gene2'], keep='first')
    TrueEdgeDict = TrueEdgeDict.iloc[:,:2]

    random_numbers = np.random.rand(len(TrueEdgeDict))
    TrueEdgeDict['weights_combined'] = random_numbers

    #保存预测的grn
    TrueEdgeDict.to_csv(os.path.join(args.output_dir,args.dataset_name,algorithm,'grn.csv'),index=False)

def run_prior_Random(args,algorithm,logger):
    logger.info(f'\n****************running prior Random**********************')
    os.makedirs(os.path.join(args.output_dir,args.dataset_name,algorithm),exist_ok=True)
    
    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)
    
    # 加载先验网络信息
    print('\nloading prior knowledge...')
    # prior_grn = pd.read_csv('./%s_prior_GRN.csv' % args.species)
    
    if args.species == 'human':
        prior_grn = pd.read_csv('/home/pengrui/CEFCON/prior_data/network_human.csv')
    else:
        prior_grn = pd.read_csv('/home/pengrui/CEFCON/prior_data/network_mouse.csv')
    
    prior_grn['from'] = prior_grn['from'].map(lambda x: x.upper())
    prior_grn['to'] = prior_grn['to'].map(lambda x: x.upper())

    # ground truth中涉及的基因个数
    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))

    #保证先验网络中的from都是TF,to都是基因表达矩阵中的基因
    prior_grn = prior_grn.loc[prior_grn['from'].isin(allNodes)
                        & prior_grn['to'].isin(allNodes), :]
    prior_grn = prior_grn.drop_duplicates(subset=['from', 'to'], keep='first')
    prior_grn = prior_grn.iloc[:,1:3]
    prior_grn.columns = ['Gene1','Gene2']

    random_numbers = np.random.rand(len(prior_grn))
    prior_grn['weights_combined'] = random_numbers

    #保存预测的grn
    prior_grn.to_csv(os.path.join(args.output_dir,args.dataset_name,algorithm,'grn.csv'),index=False)

    return True

def run_DeepSEM(args,algorithm,logger):
    logger.info(f'\n****************running DeepSEM**********************')

    save_path = os.path.join(args.output_dir,args.dataset_name,algorithm)
    os.makedirs(save_path,exist_ok=True)

    expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')
    gt_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')

    os.system(f'/home/pengrui/mambaforge/envs/grn/bin/python ./Baseline_models/DeepSEM/main.py --task non_celltype_GRN --data_file {expression_path} --net_file {gt_path} --setting new --alpha 100 --beta 1 --n_epoch 90 --save_name {save_path}')

    return True

def run_GENIE3(args,algorithm,logger):
    logger.info(f'\n****************running GENIE3**********************')

    save_path = os.path.join(args.output_dir,args.dataset_name,algorithm)
    os.makedirs(save_path,exist_ok=True)
    
    gene_expression_path = os.path.join(args.output_dir,args.dataset_name,'expression.csv')

    gt_grn_path = os.path.join(args.output_dir,args.dataset_name,'gt_grn.csv')
    gt_grn = pd.read_csv(gt_grn_path)
    # ground truth中涉及的基因个数
    allNodes = set(gt_grn.Gene1.unique()).union(set(gt_grn.Gene2.unique()))

    #加载TF列表
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

def baseline_compare(args,logger):
    # run_Random(args,'Random',logger)
    # run_NetREX(args,'NetREX',logger)
    # run_CEFCON(args,'CEFCON',logger)
    # run_Celloracle(args,'Celloracle',logger)
    # run_GRNBoost2(args,'GRNBoost2',logger)
    # run_GENIE3(args,'GENIE3',logger)


    # run_SCENIC_plus(args,'SCENIC_plus',logger)
    run_prior_Random(args,'Prior_Random',logger)

