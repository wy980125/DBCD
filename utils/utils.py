import random
import torch
import numpy as np
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from . import compute_F1
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


def eva(y_true, y_pred):
    # f1 = sample_f1_score(test_data, y_pred, num_nodes)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)

    # f1_macro, acc = cluster_acc_and_f1(y_true, y_pred)

    return nmi, ari


def compute_fast_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree, device):
    mx = max(clusters)
    MM = np.zeros((num_nodes, mx + 1))
    for i in range(len(clusters)):
        MM[i][clusters[i]] = 1
    MM = torch.tensor(MM).double().to(device)

    x = torch.matmul(torch.t(MM), torch_sparse_adj.double())
    x = torch.matmul(x, MM)
    x = torch.trace(x)

    y = torch.matmul(torch.t(MM), degree.double())
    y = torch.matmul(torch.t(y.unsqueeze(dim=0)), y.unsqueeze(dim=0))
    y = torch.trace(y)
    y = y / (2 * num_edges)
    return ((x - y) / (2 * num_edges)).item()


def compute_conductance(clusters, Graph):
    comms = [[] for i in range(max(clusters) + 1)]
    for i in range(len(clusters)):
        comms[clusters[i]].append(i)
    conductance = []
    for com in comms:
        try:
            conductance.append(nx.conductance(Graph, com, weight='weight'))
        except:
            continue

    return conductance


def evaluate_louvain(partition, labels, num_nodes, num_edges, torch_sparse_adj, degree, G, device):
    communities_louvain = partition
    FQ = compute_fast_modularity(communities_louvain, num_nodes, num_edges, torch_sparse_adj, degree, device)

    NMI, ARI = eva(labels.cpu().numpy(), communities_louvain)

    conductance = compute_conductance(communities_louvain, G)
    if len(conductance) == 0:
        avg_conductance = 0.0
    else:
        avg_conductance = sum(conductance) / len(conductance)

    y_pred = communities_louvain
    y_true = labels.cpu().numpy()
    precision = compute_F1.pairwise_precision(y_pred, y_true)
    recall = compute_F1.pairwise_recall(y_pred, y_true)
    F1 = 2 * precision * recall / (precision + recall)
    unique_clusters = len(np.unique(partition))
    result = {
        'no_of_clusters': unique_clusters,
        'modularity': round(FQ, 4),
        'avg_conductance': round(avg_conductance, 4),
        'NMI': round(NMI, 4),
        'ARI': round(ARI, 4),
        # 'f1_macro': round(f1_macro, 4),
        'F1': round(F1, 4),
        # 'ACC': round(acc, 4)
    }

    return result


def set_seed(seed=0):
    """固定所有随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimized_density(partition, adj, n_jobs=-1):
    """适用于大规模网络的社区密度计算(加权平均)

    Args:
        partition (dict): {node: community_id}格式的社区划分
        adj (sparse.csr_matrix): 稀疏邻接矩阵
        n_jobs (int): 并行任务数(-1使用全部CPU核心)

    Returns:
        float: 加权平均社区密度
    """
    # 将partition转换为社区成员字典
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # 预计算全局参数
    total_nodes = adj.shape[0]
    comm_ids = list(communities.keys())

    # 并行计算每个社区密度
    def _community_density(nodes):
        n = len(nodes)
        if n < 2:
            return 0.0
        # 稀疏矩阵高效切片
        subgraph = adj[nodes][:, nodes].tocoo()
        actual_edges = subgraph.nnz // 2  # 无向图边数修正
        max_possible = n * (n - 1)
        return actual_edges / max_possible

    densities = Parallel(n_jobs=n_jobs)(
        delayed(_community_density)(nodes)
        for nodes in communities.values()
    )

    # 计算社区权重(使用节点数加权)
    comm_sizes = np.array([len(nodes) for nodes in communities.values()])
    comm_weights = comm_sizes / comm_sizes.sum()

    return np.dot(densities, comm_weights)


def SC_score(x, labels):
    # 提取节点特征并转换为NumPy数组（确保从GPU移到CPU）
    X = x
    labels_pred = labels
    sil_score = None
    # 计算轮廓系数
    try:
        sil_score = silhouette_score(X, labels_pred)
    except ValueError as e:
        print(f"Error calculating silhouette score: {e}")
    # 处理单一社区的情况
    return round(sil_score, 4)


def CH_score(x, labels):
    # 提取节点特征并转换为NumPy数组（确保从GPU移到CPU）
    X = x
    labels_pred = labels
    ch_score = None
    # 计算轮廓系数
    try:
        ch_score = calinski_harabasz_score(X, labels_pred)
    except ValueError as e:
        print(f"Error calculating silhouette score: {e}")
    # 处理单一社区的情况
    return round(ch_score, 4)


def compute_modularity_matrix(data):
    edge_index = data.edge_index
    device = edge_index.device

    # 准确获取节点数
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else (
        torch.unique(edge_index).size(0)  # 处理非连续节点
    )

    row, col = edge_index

    # 处理无向图边数
    m = edge_index.size(1) // 2
    if m <= 0:
        raise ValueError("Invalid edge count for modularity calculation")

    # 计算度数（自动处理设备）
    k = torch.bincount(row, minlength=num_nodes).float().to(device)

    # 正确构建邻接矩阵
    A_sparse = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.size(1), device=device),
        size=(num_nodes, num_nodes)
    )
    A = A_sparse.to_dense()

    # 计算模块度矩阵
    E = torch.outer(k, k) / (2 * m)
    modularity_matrix = A - E

    return modularity_matrix, A