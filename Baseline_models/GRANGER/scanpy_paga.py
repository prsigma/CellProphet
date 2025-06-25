import pandas as pd
import scanpy as sc
import numpy as np
from models.utils import get_origin_expression_data

adata=sc.read_h5ad("XXXX.h5ad")## replace your dataset

adata.var_names_make_unique()
sc.pp.filter_cells(adata,min_genes=3) #filter cells
sc.pp.highly_variable_genes(adata, n_top_genes=100) #select top 100 highly variable genes
adata = adata[:, adata.var.highly_variable]

#data = adata.to_df() #anndata to data frame

#csv_file = 'XXX.csv'
#adata = sc.read(csv_file, cache=True)
#adata=adata.T
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
sc.tl.louvain(adata)
sc.tl.paga(adata)
adata = adata[adata.obs['louvain'].argsort()]

new_csv_file = 'ExpressionData.csv'
data_transposed = adata.copy().T
data_transposed.to_df().to_csv(new_csv_file)

def to_npy(expression_path):
    a,b=get_origin_expression_data(expression_path)
    arr = np.vstack(list(a.values()))
    np.save('time_output.npy', arr)

to_npy('ExpressionData.csv')
