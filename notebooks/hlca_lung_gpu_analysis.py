
# # RAPIDS & Scanpy Single-Cell RNA-seq Workflow


# Copyright (c) 2020, NVIDIA CORPORATION.
# 
# Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0 
# 
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


# This notebook demonstrates a single-cell RNA analysis workflow that begins with preprocessing a count matrix of size `(n_gene, n_cell)` and results in a visualization of the clustered cells for further analysis.
# For demonstration purposes, we use a dataset of ~70,000 human lung cells from Travaglini et al. 2020 (https://www.biorxiv.org/content/10.1101/742320v2) and label cells using the ACE2 and TMPRSS2 genes.


# ## Import requirements


import scanpy as sc
import anndata


import time
import os
import wget


import cudf
import cupy as cp
import numpy as np



from cuml.decomposition import PCA
from cuml.manifold import TSNE
from cuml.cluster import KMeans
from cuml.preprocessing import StandardScaler



import rapids_scanpy_funcs
import warnings
warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')


import rmm
rmm.reinitialize(
    managed_memory=False,  # Disable - causes fragmentation
    pool_allocator=True,
    initial_pool_size=int(8e9),   # 8GB fixed pool
    maximum_pool_size=int(12e9),  # No oversubscription
    devices=[0]
)


 
# We use the RAPIDS memory manager on the GPU to control how memory is allocated.

 
# ## Input data

 
# In the cell below, we provide the path to the `.h5ad` file containing the count matrix to analyze. Please see the README for instructions on how to download the dataset we use here.
# 
# We recommend saving count matrices in the sparse .h5ad format as it is much faster to load than a dense CSV file. To run this notebook using your own dataset, please see the README for instructions to convert your own count matrix into this format. Then, replace the path in the cell below with the path to your generated `.h5ad` file.


input_file = "../data/krasnow_hlca_10x.sparse.h5ad"

if not os.path.exists(input_file):
    print('Downloading import file...')
    os.makedirs('../data', exist_ok=True)
    wget.download('https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x.sparse.h5ad',
                  input_file)

 
# ## Set parameters


# marker genes
RIBO_GENE_PREFIX = "RPS" # Prefix for ribosomal genes to regress out
markers = ["ACE2", "TMPRSS2", "EPCAM"] # Marker genes for visualization

# filtering cells
min_genes_per_cell = 200 # Filter out cells with fewer genes than this expressed 
max_genes_per_cell = 6000 # Filter out cells with more genes than this expressed 

# filtering genes
min_cells_per_gene = 1 # Filter out genes expressed in fewer cells than this
n_top_genes = 5000 # Number of highly variable genes to retain

# PCA
n_components = 50 # Number of principal components to compute

# t-SNE
tsne_n_pcs = 20 # Number of principal components to use for t-SNE

# k-means
k = 35 # Number of clusters for k-means

# KNN
n_neighbors = 15 # Number of nearest neighbors for KNN graph
knn_n_pcs = 50 # Number of principal components to use for finding nearest neighbors

# UMAP
umap_min_dist = 0.3 
umap_spread = 1.0

# Gene ranking
ranking_n_top_genes = 50 # Number of differential genes to compute for each cluster


start = time.time()

 
# ## Load and Prepare Data

 
# We load the sparse count matrix from an `h5ad` file using Scanpy. The sparse count matrix will then be placed on the GPU. 


data_load_start = time.time()


%%time
adata = sc.read(input_file)
adata.X = adata.X.astype(np.float32)

 
# We maintain the index of unique genes in our dataset:


%%time
genes = cudf.Series(adata.var_names)
sparse_gpu_array = cp.sparse.csr_matrix(adata.X)

 
# Verify the shape of the resulting sparse matrix:


sparse_gpu_array.shape

 
# And the number of non-zero values in the matrix:


sparse_gpu_array.nnz


data_load_time = time.time()
print("Total data load and format time: %s" % (data_load_time-data_load_start))

 
# ## Preprocessing


preprocess_start = time.time()

 
# ### Filter

 
# We filter the count matrix to remove cells with an extreme number of genes expressed.


%%time
sparse_gpu_array = rapids_scanpy_funcs.filter_cells(sparse_gpu_array, min_genes=min_genes_per_cell, max_genes=max_genes_per_cell)

 
# Some genes will now have zero expression in all cells. We filter out such genes.


%%time
sparse_gpu_array, genes = rapids_scanpy_funcs.filter_genes(sparse_gpu_array, genes, min_cells=min_cells_per_gene)

 
# The size of our count matrix is now reduced.


sparse_gpu_array.shape

 
# ### Normalize

 
# We normalize the count matrix so that the total counts in each cell sum to 1e4.


%%time
sparse_gpu_array = rapids_scanpy_funcs.normalize_total(sparse_gpu_array, target_sum=1e4)

 
# Next, we log transform the count matrix.


%%time
sparse_gpu_array = sparse_gpu_array.log1p()

 
# ### Select Most Variable Genes

 
# We will now select the most variable genes in the dataset. However, we first save the 'raw' expression values of the ACE2 and TMPRSS2 genes to use for labeling cells afterward. We will also store the expression of an epithelial marker gene (EPCAM).


%%time
tmp_norm = sparse_gpu_array.tocsc()
marker_genes_raw = {
    ("%s_raw" % marker): tmp_norm[:, genes[genes == marker].index[0]].todense().ravel()
    for marker in markers
}

del tmp_norm

 
# We identify the top 5000 variable genes using the `cellranger` method.


%%time
hvg = rapids_scanpy_funcs.highly_variable_genes(sparse_gpu_array, genes, n_top_genes=5000)

 
# We filter the count matrix to retain only the 5000 most variable genes.


%%time
sparse_gpu_array = sparse_gpu_array[:, hvg]
genes = genes[hvg].reset_index(drop=True)
sparse_gpu_array.shape

 
# ### Regress out confounding factors (number of counts, ribosomal gene expression)

 
# We can now perform regression on the count matrix to correct for confounding factors -  for example purposes, we use the number of counts and the expression of ribosomal genes. Many workflows use the expression of mitochondrial genes (named starting with `MT-`).


ribo_genes = genes.str.startswith(RIBO_GENE_PREFIX)

 
# We now calculate the total counts and the percentage of ribosomal counts for each cell.


%%time
n_counts = sparse_gpu_array.sum(axis=1)
percent_ribo = (sparse_gpu_array[:,ribo_genes].sum(axis=1) / n_counts).ravel()

n_counts = cp.array(n_counts).ravel()
percent_ribo = cp.array(percent_ribo).ravel()

 
# And perform regression:


%%time
sparse_gpu_array = rapids_scanpy_funcs.regress_out(sparse_gpu_array, n_counts, percent_ribo)

 
# ### Scale

 
# Finally, we scale the count matrix to obtain a z-score and apply a cutoff value of 10 standard deviations, obtaining the preprocessed count matrix.


%%time
sparse_gpu_array = cp.clip(StandardScaler().fit_transform(sparse_gpu_array), a_min = -10, a_max=10)


preprocess_time = time.time()
print("Total Preprocessing time: %s" % (preprocess_time-preprocess_start))

 
# ## Cluster & Visualize

 
# We store the preprocessed count matrix as an AnnData object, which is currently in host memory. We also add the expression levels of the marker genes as observations to the annData object.


%%time

adata = anndata.AnnData(sparse_gpu_array.get())
adata.var_names = genes.to_pandas()

for name, data in marker_genes_raw.items():
    adata.obs[name] = data.get()

 
# ### Reduce

 
# We use PCA to reduce the dimensionality of the matrix to its top 50 principal components.


%%time
adata.obsm["X_pca"] = PCA(n_components=n_components, output_type="numpy").fit_transform(adata.X)

 
# ### T-SNE + k-Means

 
# We next use t-SNE on the top 20 principal components to visualize the cells in two dimensions.


%%time
adata.obsm['X_tsne'] = TSNE().fit_transform(adata.obsm["X_pca"][:,:tsne_n_pcs])

 
# We cluster the cells using k-means on the principal components. For example purposes, we set k=35.


%%time
kmeans = KMeans(n_clusters=k, random_state=0).fit(adata.obsm['X_pca'])
adata.obs['kmeans'] = kmeans.labels_.astype(str)

 
# We visualize the cells using t-SNE and label cells by color according to the k-means clustering.


%%time
sc.pl.tsne(adata, color=["kmeans"])

 
# Separately, we label the cells using the 'raw' expression of the EPCAM, ACE2 and TMPRSS2 marker genes.


%%time
sc.pl.tsne(adata, size=4, color=["ACE2_raw"], color_map="Blues", vmax=1, vmin=-0.05)
sc.pl.tsne(adata, size=4, color=["TMPRSS2_raw"], color_map="Blues", vmax=1, vmin=-0.05)
sc.pl.tsne(adata, size=4, color=["EPCAM_raw"], color_map="Reds", vmax=1,  vmin=-0.05)

 
# ### UMAP + Graph clustering

 
# We can also visualize the cells using the UMAP algorithm in Rapids. Before UMAP, we need to construct a k-nearest neighbors graph in which each cell is connected to its nearest neighbors. This can be done conveniently using rapids functionality already integrated into Scanpy.
# 
# Note that Scanpy uses an approximation to the nearest neighbors on the CPU while the GPU version performs an exact search. While both methods are known to yield useful results, some differences in the resulting visualization and clusters can be observed.


%%time
sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=knn_n_pcs, method='rapids')

 
# The UMAP function from Rapids is also integrated into Scanpy.


%%time
sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread, method='rapids')

 
# Next, we use the Louvain algorithm for graph-based clustering, once again using the `rapids` option in Scanpy.


%%time
sc.tl.louvain(adata, flavor='rapids')

 
# We plot the cells using the UMAP visualization, and using the Louvain clusters as labels.


%%time
sc.pl.umap(adata, color=["louvain"])

 
# We can also use the Leiden clustering method in RAPIDS. This method has not been integrated into Scanpy and needs to be called separately.


%%time
adata.obs['leiden'] = rapids_scanpy_funcs.leiden(adata)


%%time
sc.pl.umap(adata, color=["leiden"])

 
# We can also view cells using UMAP and labeling by raw EPCAM, ACE2 and TMPRSS2 expression.


%%time
sc.pl.umap(adata, size=4,color=["ACE2_raw"], color_map="Blues", vmax=1, vmin=-0.05)
sc.pl.umap(adata, size=4, color=["TMPRSS2_raw"], color_map="Blues", vmax=1, vmin=-0.05)
sc.pl.umap(adata, size=4, color=["EPCAM_raw"], color_map="Reds", vmax=1, vmin=-0.05)

 
# ## Differential expression analysis

 
# Once we have done clustering, we can compute a ranking for the highly differential genes in each cluster. Here we use the Louvain clusters as labels.

 
# We use logistic regression to identify the top 50 genes distinguishing each cluster.


%%time
scores, names, reference = rapids_scanpy_funcs.rank_genes_groups(adata, 
                                                                 groupby="louvain",
                                                                 n_genes=ranking_n_top_genes, 
                                                                 groups='all', 
                                                                 reference='rest')


%%time
adata.uns["rank_genes_groups"] = {}
adata.uns["rank_genes_groups"]["params"] = dict(groupby="louvain", method="logreg", reference=reference, use_raw=False)
adata.uns["rank_genes_groups"]['scores'] = scores
adata.uns["rank_genes_groups"]['names'] = names
sc.pl.rank_genes_groups(adata, n_genes=20)

 
# ## Create Zoomed View

 
# The speedup offered by Rapids makes it easy to interactively re-analyze subsets of cells. To illustrate this, we select epithelial cells (EPCAM+) from the dataset.


reanalysis_start = time.time()


%%time
epcam_cells = adata.obs["EPCAM_raw"] > 0.0
adata = adata[epcam_cells,:]
adata.shape


import rapids_singlecell as rsc
print(rsc.__version__)  # Should show 0.11.1+



%%time
adata.obsm["X_pca"] = PCA(n_components=n_components, output_type="numpy").fit_transform(adata.X)
rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=knn_n_pcs)
sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread, method='rapids')
adata.obs['leiden'] = rapids_scanpy_funcs.leiden(adata)

 
# We can repeat the dimension reduction, clustering and visualization using this subset of cells in seconds.

 
# And plot the results:


%%time
sc.pl.umap(adata, color=["leiden"])
sc.pl.umap(adata, color=["ACE2_raw"], color_map="Blues", vmax=1, vmin=-0.05)
sc.pl.umap(adata, color=["TMPRSS2_raw"], color_map="Blues", vmax=1, vmin=-0.05)


reanalysis_time = time.time()
print("Total reanalysis time : %s" % (reanalysis_time-reanalysis_start))


print("Full time: %s" % (time.time() - start))


