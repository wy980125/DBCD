from sklearn.metrics.pairwise import cosine_similarity
import torch


def merge_communities(communities, features, edge_index, num_nodes, threshold=0.85):
    """
    改进后的社区合并函数，同时满足特征相似度和边数量约束
    :param edge_index: 原始图的边索引
    :param num_nodes: 原始图的节点总数
    :param communities: 字典,communities[i]= 社区i内的节点
    :param features：社区具备的特征
    """
    # 转换为可处理格式
    comm_ids = list(communities.keys())
    feature_matrix = torch.stack([features[cid] for cid in comm_ids])

    # 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(feature_matrix.detach().cpu().numpy())

    # 构建节点到社区的映射字典（关键新增步骤）
    node_to_comm = {}
    for cid, nodes in communities.items():
        for node in nodes:
            node_to_comm[node] = cid

    # 计算社区间连边数量（关键新增逻辑）
    from collections import defaultdict
    inter_edges = defaultdict(lambda: defaultdict(int))
    for u, v in edge_index.t().tolist():
        comm_u = node_to_comm[u]
        comm_v = node_to_comm[v]
        if comm_u != comm_v:  # 仅统计跨社区边
            key = (min(comm_u, comm_v), max(comm_u, comm_v))
            inter_edges[key[0]][key[1]] += 1

    # 计算网络平均度数（关键新增计算）
    avg_degree = (edge_index.size(1)) / num_nodes

    # 并查集实现合并（修改合并条件）
    parent = {cid: cid for cid in comm_ids}

    def find(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    # 双重条件合并逻辑（核心修改部分）
    n = len(comm_ids)
    for i in range(n):
        for j in range(i + 1, n):
            cid1, cid2 = comm_ids[i], comm_ids[j]
            # 获取社区对的标准形式
            a, b = (cid1, cid2) if cid1 < cid2 else (cid2, cid1)

            # 同时满足两个条件
            if (sim_matrix[i][j] > threshold and
                    inter_edges[a].get(b, 0) > avg_degree):
                root1 = find(cid1)
                root2 = find(cid2)
                if root1 != root2:
                    parent[root2] = root1

    # 重新组织合并后的社区
    merged_communities = defaultdict(list)
    for cid in comm_ids:
        root = find(cid)
        merged_communities[root].extend(communities[cid])

    return merged_communities


