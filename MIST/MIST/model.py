import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

    def forward(self, c, h_pl, h_mi):
        c_x = torch.unsqueeze(c, 0).expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        return torch.cat((sc_1, sc_2), 0)

class ClusterLayer(nn.Module):
    def __init__(self, n_clusters: int, dim: int, alpha: float = 1.0):
        super().__init__()
        self.clusters = nn.Parameter(torch.Tensor(int(n_clusters), int(dim)))
        nn.init.xavier_uniform_(self.clusters)
        self.alpha = alpha

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dist = torch.sum((z.unsqueeze(1) - self.clusters.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        return q / torch.sum(q, dim=1, keepdim=True).clamp_min(1e-12)

class AdaptiveFusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    def forward(self, h_gene, h_spat):
        g = self.gate(torch.cat([h_gene, h_spat], dim=-1))
        return g * h_gene + (1 - g) * h_spat

class MIST(nn.Module):
    def __init__(self, dim_in, dim_out, num_scales=3, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        self.dropout = dropout
        
        self.gene_proj = nn.Linear(dim_in, dim_out)
        self.spat_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, dim_out)
        )
        
        self.fusion_gate = AdaptiveFusionGate(dim_out)
        
        self.gnn_weights = nn.ModuleList([
            nn.Linear(dim_out, dim_out, bias=False) for _ in range(num_scales)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim_out) for _ in range(num_scales)])
        self.decoder = nn.Linear(dim_out, dim_in)
        self.act = F.elu

    def forward(self, x_gene, x_spat, adjs):
        h_gene = self.act(self.gene_proj(x_gene))
        h_spat = self.act(self.spat_proj(x_spat))
        
        h_init = self.fusion_gate(h_gene, h_spat)
        
        scale_outputs = []
        for i in range(self.num_scales):
            h = F.dropout(h_init, self.dropout, training=self.training)
            h = self.gnn_weights[i](h)
            h = torch.mm(adjs[i], h)
            h = self.bns[i](h)
            scale_outputs.append(self.act(h))

        h_avg = sum(scale_outputs) / self.num_scales
        emb = 0.7*h_avg + 0.3 * h_init 
        recon = self.decoder(emb)
        return emb, recon, scale_outputs