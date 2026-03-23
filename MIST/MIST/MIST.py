import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from .model import MIST, ClusterLayer, Discriminator
from .preprocess import preprocess_adj,fix_seed

class MIST:
    def __init__(self, adata, device=torch.device('cuda:1'), epochs=600, warmup_epochs=200, 
                 beta=2.0, alpha=1.0, lamda=1.0, num_scales=3,seed = 42):
        self.adata = adata
        self.device = device
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.beta, self.alpha, self.lamda = beta, alpha, lamda
        self.num_scales = num_scales
        self.dim_in = adata.obsm['feat'].shape[1]
        self.dim_out = 64
        self.random_seed = seed

    def _initialize(self, n_clusters):
        self.model = MIST(self.dim_in, self.dim_out, num_scales=self.num_scales).to(self.device)
        self.cluster_layer = ClusterLayer(n_clusters, self.dim_out).to(self.device)
        self.disc = Discriminator(self.dim_out).to(self.device)
        self.as_norm = [torch.FloatTensor(preprocess_adj(self.adata.obsm[f'adj_scale_{i}'])).to(self.device) 
                        for i in range(self.num_scales)]
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.cluster_layer.parameters()) + list(self.disc.parameters()),
            lr=0.001, weight_decay=1e-4
        )

    def train(self, n_clusters=7):
        
        fix_seed(self.random_seed)
        self._initialize(n_clusters)    
        feat = torch.FloatTensor(self.adata.obsm['feat']).to(self.device)
        feat_a = torch.FloatTensor(self.adata.obsm['feat_a']).to(self.device)
        spat = torch.FloatTensor(self.adata.obsm['spatial_norm']).to(self.device)
        lbl = torch.cat([torch.ones(feat.shape[0]), torch.zeros(feat.shape[0])]).to(self.device)
        
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()
            
            emb, recon, scale_feats = self.model(feat, spat, self.as_norm)
            emb_a, _, _ = self.model(feat_a, spat, self.as_norm)

            loss_recon = F.mse_loss(feat, recon)
            
            summary = torch.sigmoid(torch.mean(emb, dim=0))
            logits = self.disc(summary, emb, emb_a)
            loss_cl = F.binary_cross_entropy_with_logits(logits, lbl)
            
            loss_consist = sum(F.mse_loss(scale_feats[i], scale_feats[i+1]) for i in range(len(scale_feats)-1))
            
            loss = self.beta * loss_recon + self.lamda * loss_cl + 0.1 * loss_consist 


            if epoch >= self.warmup_epochs:
                if epoch == self.warmup_epochs:
                    km = KMeans(n_clusters=n_clusters, n_init=20,random_state=self.random_seed).fit(emb.detach().cpu().numpy())
                    self.cluster_layer.clusters.data.copy_(torch.tensor(km.cluster_centers_).to(self.device))
                
                q = self.cluster_layer(emb)
                p = ((q**2 / q.sum(0)).t() / (q**2 / q.sum(0)).sum(1)).t().detach()
                loss_kl = F.kl_div(q.log(), p, reduction='batchmean')
                loss += self.alpha * loss_kl

            loss.backward()
            self.optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")

        self.model.eval()
        with torch.no_grad():
            emb_tensor, _, scale_feats = self.model(feat, spat, self.as_norm)
            
            self.adata.obsm['emb'] = emb_tensor.detach().cpu().numpy()

        return self.adata

    def process(self, n_clusters=7, refinement=True, radius=30):
        from .utils import clustering
        clustering(self.adata, n_clusters=n_clusters, radius=radius, key='emb', refinement=refinement)
        return self.adata