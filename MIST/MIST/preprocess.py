import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import torch

def preprocess(adata, n_top_genes=3000, seed=42):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    #sc.pp.normalize_total(adata, target_sum=1e4)
    #sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50,random_state = seed)
    
    adata.obsm['feat'] = adata.obsm['X_pca'].astype(np.float32)
    coords = adata.obsm['spatial'].astype(np.float32)
    adata.obsm['spatial_norm'] = (coords - coords.min(0)) / (coords.max(0) - coords.min(0))
    
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(adata.n_obs))
    adata.obsm['feat_a'] = adata.obsm['feat'][ids]

def construct_gated_interaction(adata, n_neighbors_list=[3, 10, 20]):

    feat = adata.obsm['feat']
    pos = adata.obsm['spatial']
    n_spot = pos.shape[0]
    
    for scale_idx, n_nb in enumerate(n_neighbors_list):
        print(f"Building Scale {scale_idx} (K={n_nb}) with Soft-Weights...")
        nbrs = NearestNeighbors(n_neighbors=n_nb + 1 ,algorithm='brute').fit(pos)
        _, indices = nbrs.kneighbors(pos)
        
        rows, cols, weights = [], [], []
        for i in range(n_spot):

            neighbor_idx = indices[i, 1:]

            sims = cosine_similarity(feat[i:i+1], feat[neighbor_idx])[0]
            tau = 0.2   
            sims = np.exp(sims / tau) / np.sum(np.exp(sims / tau)) 

            for j, sim in enumerate(sims):

                if sim > 0:
                    rows.append(i)
                    cols.append(neighbor_idx[j])
                    weights.append(sim) 
        
        adj = sp.csr_matrix((weights, (rows, cols)), shape=(n_spot, n_spot))
        adata.obsm[f'adj_scale_{scale_idx}'] = adj.toarray()

def preprocess_adj(adj):

    adj = sp.coo_matrix(adj) + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return adj.dot(sp.diags(d_inv_sqrt)).transpose().dot(sp.diags(d_inv_sqrt)).toarray()

def fix_seed(seed=42):
    import os, random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)