import networkx as nx
import community as community_louvain
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch


# 生成一个随机图，并附加随机属性
def generate_graph(n=24, m=120, attr_dim=5):
    G = nx.gnm_random_graph(n, m, seed=42)
    np.random.seed(42)
    for i in G.nodes():
        G.nodes[i]['attr'] = np.random.rand(attr_dim)
    return G


# 1. Louvain 算法进行结构社区检测
def detect_louvain_communities(G):
    partition = community_louvain.best_partition(G, random_state=0)
    return partition


# 2. K-Means 进行属性聚类
def detect_kmeans_communities(G, n_clusters):
    attributes = np.array([G.nodes[i]['attr'] for i in G.nodes()])
    attributes = StandardScaler().fit_transform(attributes)  # 归一化
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(attributes)
    return {node: labels[i] for i, node in enumerate(G.nodes())}


def detect_kmeans_communities_data(G, n_clusters, data):
    # 获取节点属性，假设 data.x 是一个 PyTorch Tensor
    attributes = data.x.cpu().numpy() # 将 Tensor 转换为 NumPy 数组
    #attributes = StandardScaler().fit_transform(attributes)  # 归一化
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(attributes)

    # 返回每个节点的社区标签
    return {node: labels[i] for i, node in enumerate(G.nodes())}
# 3. 筛选大社区
def filter_large_communities(comm_A):
    from collections import Counter
    community_sizes = Counter(comm_A.values())
    sizes = list(community_sizes.values())

    if not sizes:
        return {}

    # 计算动态阈值
    mean_size = np.mean(sizes)
    #std_size = np.std(sizes)
    threshold = mean_size  # 核心修改点

    # 筛选大型社区
    large_communities = {c for c, size in community_sizes.items() if size > threshold}
    return {node: comm for node, comm in comm_A.items() if comm in large_communities}


# 3. 根据社区内部的平均聚类系数进行社区过滤
def filter_large_communities_CC(G, comm_A):
    from collections import defaultdict

    # 计算每个节点的局部聚类系数
    clustering_coeffs = nx.clustering(G)

    # 计算每个社区的平均聚类系数
    community_clustering = defaultdict(list)
    for node, comm in comm_A.items():
        community_clustering[comm].append(clustering_coeffs[node])

    community_avg_clustering = {comm: np.mean(coeffs) for comm, coeffs in community_clustering.items()}

    # 计算所有社区的平均聚类系数及其标准差
    avg_clustering = np.mean(list(community_avg_clustering.values()))
    std_clustering = np.std(list(community_avg_clustering.values()))

    # 设定阈值：平均值 + 0.5 * 标准差
    threshold = avg_clustering + 0.5 * std_clustering

    # 过滤符合条件的社区
    selected_communities = {c for c, avg_c in community_avg_clustering.items() if avg_c > threshold}

    return {node: comm for node, comm in comm_A.items() if comm in selected_communities}


# 4. 计算共识矩阵（不限制边，仅考虑筛选出的大社区）
def compute_consensus_matrix(filtered_nodes, comm_A, comm_B):
    node_list = sorted(filtered_nodes)  # 只考虑筛选出的节点，并保持索引一致
    node_index = {node: i for i, node in enumerate(node_list)}

    size = len(node_list)
    consensus_matrix = np.zeros((size, size))

    for i, u in enumerate(node_list):
        for j, v in enumerate(node_list):
            if i < j and comm_A[u] == comm_A[v] and comm_B[u] == comm_B[v]:
                consensus_matrix[i, j] = 1
                consensus_matrix[j, i] = 1  # 确保对称

    return consensus_matrix, node_index


def build_joint_onehot(C_l, C_f):
    """
    构造联合 one-hot 矩阵，使得 M = C_joint @ C_joint.T

    Args:
        C_l: list[int], 结构聚类标签
        C_f: list[int], 特征聚类标签
    Returns:
        C_joint: torch.Tensor of shape (n', k'), 联合 one-hot 编码
    """
    assert len(C_l) == len(C_f), "结构聚类和特征聚类长度不一致"
    n = len(C_l)

    # 构造联合标签元组
    joint_labels = list(zip(C_l, C_f))  # [(a1, b1), (a2, b2), ...]

    # 映射为唯一ID
    unique_pairs = list(sorted(set(joint_labels)))  # 所有不同组合 (结构, 特征)
    pair_to_id = {pair: idx for idx, pair in enumerate(unique_pairs)}  # 映射到 0...k'-1

    # 构造 one-hot 矩阵
    p = len(unique_pairs)
    C_joint = torch.zeros((n, p))

    for i, pair in enumerate(joint_labels):
        j = pair_to_id[pair]
        C_joint[i, j] = 1.0

    return C_joint


def build_joint_onehot_from_labels(filtered_nodes, comm_A, comm_B):
    node_list = sorted(filtered_nodes)
    C_l = [comm_A[v] for v in node_list]
    C_f = [comm_B[v] for v in node_list]
    return build_joint_onehot(C_l, C_f), node_list
# # 测试代码
# G = generate_graph()
# comm_A = detect_louvain_communities(G)
# n_clusters = len(set(comm_A.values()))  # 使用 Louvain 社区数作为 K-means 聚类数
# comm_B = detect_kmeans_communities(G, n_clusters)
# filtered_comm_A = filter_large_communities(comm_A)  # 筛选大社区
# filtered_nodes = set(filtered_comm_A.keys())  # 筛选出的节点集
# print(filtered_nodes)
# consensus_matrix, node_index = compute_consensus_matrix(filtered_nodes, filtered_comm_A, comm_B)
#
# print("共识矩阵：")
# print(consensus_matrix)
# print("节点索引映射：")
# print(node_index)
#
# consensus_matrix_tensor = torch.tensor(consensus_matrix, dtype=torch.float32)
# # 假设 embeddings 是形状为 (num_nodes, embedding_dim) 的 PyTorch 张量
# # 需要获取 node_index 对应的行
# embeddings = torch.randn(24, 6)
# filtered_indices = torch.tensor(list(node_index.keys()), dtype=torch.long)  # 获取索引列表
# filtered_embeddings = embeddings[filtered_indices]  # 直接索引提取
# embedding_similarity = filtered_embeddings @ filtered_embeddings.T  # (num_filtered_nodes, num_filtered_nodes)
# mse_loss = torch.nn.functional.mse_loss(embedding_similarity, consensus_matrix_tensor)
#
# print("MSE Loss:", mse_loss.item())
#
#
# filtered_comm_A_CC = filter_large_communities_CC(G, comm_A)
#
# filtered_nodes_CC = set(filtered_comm_A_CC.keys())
#
# print(filtered_nodes_CC)
# consensus_matrix_CC, node_index_CC = compute_consensus_matrix(filtered_nodes_CC, filtered_comm_A_CC, comm_B)
#
# print("_CC共识矩阵：")
# print(consensus_matrix_CC)
# print("_CC节点索引映射：")
# print(node_index_CC)
# filtered_indices_CC = torch.tensor(list(node_index_CC.keys()), dtype=torch.long)
#
# consensus_matrix_tensor_CC = torch.tensor(consensus_matrix_CC, dtype=torch.float32)
#
# filtered_embeddings_CC = embeddings[filtered_indices_CC]  # 直接索引提取
# embedding_similarity_CC = filtered_embeddings_CC @ filtered_embeddings_CC.T  # (
# mse_loss_CC = torch.nn.functional.mse_loss(embedding_similarity_CC, consensus_matrix_tensor_CC)
#
# print("MSE Loss_CC:", mse_loss_CC.item())
def compute_onehot_from_consensus(consensus_matrix, node_index):
    size = len(node_index)
    labels = -np.ones(size, dtype=int)  # 初始化每个节点的社区标签为 -1
    community_id = 0  # 社区编号

    # 遍历共识矩阵，将每个连通分量分配给一个社区
    for i in range(size):
        if labels[i] == -1:  # 该节点尚未被分配社区
            queue = [i]
            labels[i] = community_id  # 赋予当前社区编号

            while queue:
                node = queue.pop(0)
                for neighbor in range(size):
                    if consensus_matrix[node, neighbor] == 1 and labels[neighbor] == -1:
                        labels[neighbor] = community_id
                        queue.append(neighbor)

            community_id += 1  # 切换到下一个社区编号

    # 生成 one-hot 矩阵
    onehot_matrix = np.zeros((size, community_id))
    for i in range(size):
        onehot_matrix[i, labels[i]] = 1

    return onehot_matrix, labels
