import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.distributions.normal import Normal

class GCNEncoder(nn.Module):
    """
    Paper Eq (10): Encoder q(Z | X, A)
    Uses a simple two-layer GCN.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Define two GCN layers (conv1, conv2_mu, conv2_logstd)
        # self.conv1 = ...
        # self.conv_mu = ...
        # self.conv_logstd = ...
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (N+1, D) (Attributes + Task)
            edge_index: Anchor graph connectivity
        Returns:
            mu, log_std
        """
        # TODO: Implement forward pass
        # h = F.relu(self.conv1(x, edge_index))
        # mu = ...
        # log_std = ...
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        mu = self.conv_mu(h, edge_index)
        log_std = self.conv_logstd(h, edge_index)
        # Placeholder return
        return mu, log_std

class GraphDecoder(nn.Module):
    """
    Paper Eq (12) & (13):
    1. Sketching: p_s(S | Z) -> Dense Graph
    2. Refining: p_c(G | S) -> Sparse Graph via Regularization
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        # This corresponds to the FFN_d in Eq (13)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim * 3, 64), # [h_i, h_j, h_task]
            nn.ReLU(),
            nn.Linear(64, 1) # scalar output w_ij
        )
        self.temperature = 1.0

    def forward(self, z):
        """
        Args:
            z: Latent representations Z from Encoder (N+1, latent_dim)
               Note: The last row of z is usually h_task
        Returns:
            adj_prob: Predicted Adjacency Matrix S (N, N)
        """
        # TODO: Implement the pairwise sketching logic
        # 1. Separate agent nodes (h_1...h_N) and task node (h_task)
        # 2. Compute w_ij for all pairs using self.ffn([h_i, h_j, h_task])
        # 3. Apply Sigmoid with temperature (Eq 13)
        agent_nodes = z[:-1]
        task_node = z[-1:]
        # 2. 构造所有配对 (i, j) 
        # 我们要造出两个大矩阵，形状都是 (N, N, D)
        # hi_chart: 第 (i, j) 格子放的是 h_i
        # hj_chart: 第 (i, j) 格子放的是 h_j
        N = agent_nodes.size(0)
        hi_chart = agent_nodes.unsqueeze(1).repeat(1, N, 1)
        hj_chart = agent_nodes.unsqueeze(0).repeat(N, 1, 1)
        task_repeated = task_node.unsqueeze(0).repeat(N, N, 1)
        combined = torch.cat([hi_chart, hj_chart, task_repeated], dim=-1)
        w = self.ffn(combined).squeeze(-1)
        if self.training:
            epsilon = torch.rand_like(w) + 1e-8
            gumbel_noise = torch.log(epsilon) - torch.log(1 - epsilon)
            s = torch.sigmoid((w + gumbel_noise) / self.temperature)
        else:
            s = torch.sigmoid(w)
        
        return s

class VGAE(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim)
        
        # Learnable sparsity parameters (Eq 14)
        # Z and W matrices for S = Z W Z^T decomposition
        self.r = 3
        self.Z = None # Will be initialized dynamically or fixed size
        self.W = nn.Parameter(torch.randn(self.r, self.r))

    def reparameterize(self, mu, log_std):
        """
        Standard VAE reparameterization trick: z = mu + sigma * epsilon
        """
        if self.training:
            # TODO: Implement sampling
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x_agent, x_task, anchor_edge_index):
        """
        Full pass:
        1. Construct Graph: Combine agents and task node.
        2. Encode: Get latent Z.
        3. Decode: Get Adjacency Matrix S.
        """
        # 1. Construction (Eq 8)
        # Concatenate task node to the end of agents
        # x_full = torch.cat([x_agent, x_task], dim=0)
        
        # 2. Encoding
        # mu, log_std = self.encoder(x_full, anchor_edge_index)
        # z = self.reparameterize(mu, log_std)
        
        # 3. Decoding
        # adj = self.decoder(z)

        x_full = torch.cat([x_agent, x_task], dim=0)

        mu, log_std = self.encoder(x_full, anchor_edge_index)
        z = self.reparameterize(mu, log_std)

        adj = self.decoder(z)
        
        return adj

    def compute_loss(self, S_pred, anchor_adj):
        """
        Input:
            S_pred: (N, N) Decoder输出的邻接矩阵
            anchor_adj: (N, N) 锚点图
        """
        # 1. Sparsity Loss (论文核心创新点)
        # S = U \Sigma V^T
        # Z = U[:, :self.r] (Top-r 左奇异向量)
        U, S_val, V = torch.linalg.svd(S_pred)
        Z = U[:, :self.r]
        
        # Loss = |S - Z W Z^T|^2
        S_reconstructed = Z @ self.W @ Z.t()
        loss_sparsity = torch.norm(S_pred - S_reconstructed, p='fro')
        
        # 2. Anchor Loss
        loss_anchor = torch.norm(anchor_adj - S_reconstructed, p='fro')
        
        # 3. W 正则 (Nuclear Norm 最小化 -> 稀疏化)
        # 即使只用了 r 个，也可以约束 W 的核范数
        loss_W = torch.sum(torch.linalg.svd(self.W).S) # 核范数 = 奇异值之和
        
        return loss_sparsity + loss_anchor + loss_W
