import random
import torch
import torch.nn.functional as F
import numpy as np


def loss(Z, adj, num_edges):
    alpha = 1.0

    # --------自旋玻璃版本loss-----
    # tr_A = torch.trace(S.T @ s_adj_batch.to(dtype=torch.float32) @ S)
    # # tr_A = torch.trace(S.T @ s_adj_batch @ S)
    # community_sum = scatter_add(S, torch.zeros(S.size(0), dtype=torch.long, device=S.device), dim=0)
    # tr_ones = torch.sum(community_sum ** 2)
    # --------自旋玻璃版本loss-----

    # --------高阶Q版本loss-----
    # 计算 Z^T * s_adj * Z，利用稀疏矩阵乘法
    # Z = S
    # s_adj = s_adj_batch
    # GAMMA = 1 / num_batch
    # ZT = Z.T  # K x N
    # ZT_s_adj = torch.sparse.mm(s_adj, Z)  # N x K
    # term1 = torch.sum(ZT @ ZT_s_adj)  # 计算迹
    #
    # # 计算 tr(GAMMA * Z^T * Z) = GAMMA * ||Z||_F^2
    # term2 = GAMMA * torch.sum(Z ** 2)
    #
    # Q_loss = term1 - term2
    # print(Q_loss.item())
    # --------高阶Q版本loss-----

    # --------高阶Q版本loss稠密矩阵-----

    s_adj = adj
    Q_loss = torch.trace(Z.T @ s_adj @ Z)
    # Q_loss = torch.trace(s_adj @ Z @ Z.T)
    # print(Q_loss.item())
    # --------高阶Q版本loss稠密矩阵-----

    # # # --------重构损失稠密矩阵-----
    # A_coo = A.tocoo()
    # row, col = A_coo.row, A_coo.col  # 获取邻接矩阵的非零索引
    # # 获取邻接矩阵的非零索引 (row, col)
    #
    # # 计算 Z_i^T * Z_j
    # ZZT_values = (Z[row] * Z[col]).sum(dim=1)  # 计算点积
    #
    # # 计算 Sigmoid (σ(Z_i^T Z_j))
    # preds = torch.sigmoid(ZZT_values)  # 预测边的概率
    # targets = torch.ones_like(preds)  # A_ij = 1 (原始图中的边)
    #
    # # 计算二元交叉熵 (BCE) 误差
    # con_loss = F.binary_cross_entropy(preds, targets)  # 计算损失
    # # --------重构损失稠密矩阵-----

    # # 新增崩溃损失计算
    # cluster_sizes = S.sum(dim=0)  # 每个簇的节点数 [num_communities]
    # n_nodes = S.size(0)  # 总节点数
    # l2_norm = torch.norm(cluster_sizes, p=2)  # L2范数
    # sqrt_k = torch.sqrt(torch.tensor(  # 自动处理设备
    #     self.conv2.out_channels,
    #     dtype=torch.float32,
    #     device=S.device
    # ))
    #
    # collapse_loss = (l2_norm / n_nodes) * sqrt_k - 1.0
    #num_edges = edge_index.size(1)
    # 合并总损失
    # total_loss = -((1 + self.gamma) * tr_A - self.gamma * tr_ones) + self.collapse_reg * collapse_loss
    # total_loss = -alpha * Q_loss / num_edges + self.collapse_reg * collapse_loss + 0.01 * con_loss
    total_loss = -alpha * Q_loss / num_edges
    return total_loss


# print(self.gamma.item())

def virtual_node_loss(Z, virtual_ids, merged_communities):
    """
    计算虚拟节点（社区中心）与其社区内节点的点积损失（基于归一化嵌入）

    :param Z: 归一化后的嵌入向量 (num_nodes, embedding_dim)
    :param virtual_ids: 虚拟节点的索引 (num_virtual,)
    :param merged_communities: 字典 {社区ID: [社区内的节点索引]}
    :return: 计算出的损失值
    """
    loss = 0.0
    count = 0

    for i, (cid, nodes) in enumerate(merged_communities.items()):
        virtual_idx = virtual_ids[i]  # 当前社区的虚拟节点索引
        real_nodes = torch.tensor(nodes, dtype=torch.long, device=Z.device)  # 该社区的实际节点索引

        if len(real_nodes) == 0:
            continue  # 跳过空社区

        # 获取虚拟节点的嵌入
        virtual_embedding = Z[virtual_idx]  # (embedding_dim,)

        # 获取该社区内的所有节点的嵌入
        real_embeddings = Z[real_nodes]  # (num_nodes_in_community, embedding_dim)

        # 计算点积（等价于归一化后的余弦相似度）
        similarity = (real_embeddings @ virtual_embedding).mean()  # 取均值

        # 余弦相似度最大化等价于最小化 (1 - similarity)
        loss += (1 - similarity)
        count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=Z.device)


def convert_scipy_torch_sp(sp_adj):
    sp_adj = sp_adj.tocoo()
    indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
    sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
    return sp_adj


def Q_loss(output, num_nodes, num_edges, sparse_adj, degree, device):
    sample_size = int(1 * num_nodes)
    s = random.sample(range(0, num_nodes), sample_size)

    s_output = output[s, :]
    # s2 = s
    s_adj = sparse_adj[s, :][:, s]
    s_adj = convert_scipy_torch_sp(s_adj)
    s_degree = degree[s]
    # print(s_degree)

    x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(device))
    x = torch.matmul(x, s_output.double())
    x = torch.trace(x)

    y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(device))
    y = (y ** 2).sum()
    y = y / (2 * num_edges)

    scaling = 1.0
    #scaling = num_nodes ** 2 / (sample_size ** 2)

    m_loss = -((x - y) / (2 * num_edges)) * scaling
    return m_loss


def f_loss(out, C):
    sample_size = out.shape[0]
    #print(sample_size)
    t1 = torch.matmul(torch.t(C), C)
    t1 = torch.matmul(t1, t1)
    t1 = torch.trace(t1)

    t2 = torch.matmul(torch.t(out), out)
    t2 = torch.matmul(t2, t2)
    t2 = torch.trace(t2)

    t3 = torch.matmul(torch.t(out), C)
    t3 = torch.matmul(t3, torch.t(t3))
    t3 = torch.trace(t3)

    aux_objective_loss = 1 / (sample_size ** 2) * (t1 + t2 - 2 * t3)
    #print(aux_objective_loss)
    return aux_objective_loss