from utils import load_data, args, utils
import os
import torch
# import scipy.sparse as sp
import numpy as np
import re
from torch_geometric.data import Data
# import scipy as sp
from torch_geometric.utils import to_undirected
import scipy.sparse as sp
import networkx as nx


def load_graph_data(root_path=".", dataset_name="dblp"):
    load_path = root_path + "/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path + "_feat.npy", allow_pickle=True)
    label = np.load(load_path + "_label.npy", allow_pickle=True)
    adj = np.load(load_path + "_adj.npy", allow_pickle=True)
    # 强制转换特征矩阵为float32
    feat = np.asarray(feat, dtype=np.float32)  # 新增类型转换

    # 检查是否仍存在字符串类型
    if feat.dtype == np.dtype('O'):  # 'O'表示对象类型（可能包含字符串）
        raise ValueError(f"特征矩阵包含非数值数据，请检查{dataset_name}的数据源")
    return feat, label, adj


def uni_load_data(dataset, device):
    # 根据数据集类型选择不同的加载方式
    if dataset in ['cora', 'cite', 'pubmed', 'computers', 'photo',
                   'coauthorcs', 'coauthorphysics', 'ogbg-molhiv', 'nell']:
        # 第一类数据集加载方式
        data = load_data.load_data(dataset).to(device)
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_edges = edge_index.shape[1] // 2
        labels = data.y.flatten()

        # 创建稀疏邻接矩阵
        torch_sparse_adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(num_edges * 2).to(device),
            size=(num_nodes, num_nodes)
        )

        # 转换为scipy稀疏矩阵
        sparse_adj = sp.csr_matrix(
            (np.ones(edge_index.shape[1]), edge_index.cpu().numpy()),
            shape=(num_nodes, num_nodes)
        )
    elif dataset in ['dblp', 'acm', 'amap', 'corafull', 'texas', 'wisc', 'bat', 'wiki']:
        # 第二类数据集加载方式
        root_path = './data'
        feat, label, adj = load_graph_data(root_path, dataset)

        # 转换为PyTorch张量
        feat = torch.tensor(feat, dtype=torch.float32).to(device)
        labels = torch.tensor(label, dtype=torch.long).flatten().to(device)
        num_nodes = adj.shape[0]

        # 构建edge_index
        sparse_adj = sp.csr_matrix(adj)
        coo_adj = sparse_adj.tocoo()
        edge_index = torch.tensor(
            np.vstack((coo_adj.row, coo_adj.col)),
            dtype=torch.long
        ).to(device)

        # 创建Data对象
        data = Data(x=feat, edge_index=edge_index, y=labels)
        num_edges = sparse_adj.nnz // 2  # 无向图边数

        # 创建PyTorch稀疏邻接矩阵
        torch_sparse_adj = torch.sparse_coo_tensor(
            edge_index,
            torch.tensor(coo_adj.data, dtype=torch.float32).to(device),
            size=(num_nodes, num_nodes)
        )


    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return data, edge_index, num_nodes, num_edges, labels, sparse_adj, torch_sparse_adj


def load_LFR_network(datasetname, device, binary_feature=False, EA=False):
    """
    加载指定名称的 LFR 网络数据并返回相应的边索引、节点数和标签

    :param datasetname: 数据集名称，例如 'LFR_n1000_tau1-3.0_tau2-1.5_mu-0.7_hunxiao-0.4'
    :param device: 目标设备（例如 'cuda' 或 'cpu'）
    :return: edge_index, num_nodes, num_edges, labels, torch_sparse_adj, sparse_adj
    """
    # 解析 datasetname 获取网络参数
    parts = datasetname.split('_')
    if EA:
        parts = datasetname.split('_')
        n = int(parts[1][1:])  # 节点数 n
        tau1 = float(parts[2].split('-')[1])  # Tau1参数
        tau2 = float(parts[3].split('-')[1])  # Tau2参数

        mu = float(parts[4].split('-')[1])  # mu参数
        num_features = int(parts[6].split('-')[1])  # hx参数
        dom = int(parts[7].split('-')[1])
        noise = float(parts[8].split('-')[1])
        file_path = f"data/LFR/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_num_features-{num_features}_dom-{dom}_noise-{noise}/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_mu-{mu}_num_features-{num_features}_dom-{dom}_noise-{noise}.pt"
    else:
        n = int(parts[1][1:])  # 节点数 n
        tau1 = float(parts[2].split('-')[1])  # Tau1参数
        tau2 = float(parts[3].split('-')[1])  # Tau2参数

        mu = float(parts[4].split('-')[1])  # mu参数
        hx = float(parts[5].split('-')[1])  # hx参数
        if binary_feature:
            file_path = f"data/LFR/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_hx-{hx}_01/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_mu-{mu}_hx-{hx}_01.pt"
        else:
            # 构建文件路径
            file_path = f"data/LFR/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_hx-{hx}/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_mu-{mu}_hunxiao-{hx}.pt"



    # 加载保存的 LFR 网络

    data = torch.load(file_path, map_location=device)

    # 假设 LFR 网络包含 'edge_index' 和 'labels'，可以根据实际存储内容调整
    edge_index = data.edge_index  # 假设数据字典中包含 edge_index
    labels = data.y.flatten()  # 假设数据字典中包含标签（如果存在）
    num_nodes = data.num_nodes  # 节点数量
    # edge_index = to_undirected(edge_index)
    # 计算边的数量
    num_edges = edge_index.shape[1] // 2

    # 创建稀疏邻接矩阵
    torch_sparse_adj = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.shape[1]).to(device),
        size=(num_nodes, num_nodes)
    )

    # 转换为scipy稀疏矩阵
    sparse_adj = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), edge_index.cpu().numpy()),
        shape=(num_nodes, num_nodes)
    )

    return data, edge_index, num_nodes, num_edges, labels, torch_sparse_adj, sparse_adj


def save_LFR_network_numpy(datasetname, device, binary_feature=False, EA=False):
    """
    加载指定名称的 LFR 网络数据并返回相应的边索引、节点数和标签

    :param datasetname: 数据集名称，例如 'LFR_n1000_tau1-3.0_tau2-1.5_mu-0.7_hunxiao-0.4'
    :param device: 目标设备（例如 'cuda' 或 'cpu'）
    :return: edge_index, num_nodes, num_edges, labels, torch_sparse_adj, sparse_adj
    """
    parts = datasetname.split('_')
    if EA:
        parts = datasetname.split('_')
        n = int(parts[1][1:])  # 节点数 n
        tau1 = float(parts[2].split('-')[1])  # Tau1参数
        tau2 = float(parts[3].split('-')[1])  # Tau2参数

        mu = float(parts[4].split('-')[1])  # mu参数
        num_features = int(parts[6].split('-')[1])  # hx参数
        dom = int(parts[7].split('-')[1])
        noise = float(parts[8].split('-')[1])
        file_path = f"data/LFR/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_num_features-{num_features}_dom-{dom}_noise-{noise}/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_mu-{mu}_num_features-{num_features}_dom-{dom}_noise-{noise}.pt"
    else:
        n = int(parts[1][1:])  # 节点数 n
        tau1 = float(parts[2].split('-')[1])  # Tau1参数
        tau2 = float(parts[3].split('-')[1])  # Tau2参数

        mu = float(parts[4].split('-')[1])  # mu参数
        hx = float(parts[5].split('-')[1])  # hx参数
        if binary_feature:
            file_path = f"data/LFR/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_hx-{hx}_01/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_mu-{mu}_hx-{hx}_01.pt"
        else:
            # 构建文件路径
            file_path = f"data/LFR/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_hx-{hx}/LFR_n{n}_tau1-{tau1}_tau2-{tau2}_mu-{mu}_hunxiao-{hx}.pt"
    # 加载保存的 LFR 网络
    data = torch.load(file_path, map_location=device)
    G = load_data.get_graph(data)  # 假设该函数返回 networkx 图对象

    # ================== 新增：生成并保存 numpy 数据 ==================
    # 构建保存路径 (例如：data/LFR/datasetname/datasetname)
    save_dir = os.path.dirname(file_path)  # 获取父目录，如 data/LFR/LFR_n1000_...
    save_dir = os.path.join(save_dir, datasetname)
    os.makedirs(save_dir, exist_ok=True)  # 创建新文件夹
    # base_name = os.path.splitext(os.path.basename(file_path))[0]  # 去除 .pt 的文件名
    save_path = os.path.join(save_dir, datasetname)  # 完整保存路径前缀

    # 确保目录存在
    # os.makedirs(save_dir, exist_ok=True)

    # 1. 保存邻接矩阵 (从 networkx 图生成)
    adj = nx.adjacency_matrix(G, weight=None).astype(np.float32)  # 得到 scipy 稀疏矩阵
    adj_dense = adj.toarray()  # 转换为密集 numpy 数组
    np.save(save_path + "_adj.npy", adj_dense)

    # 2. 保存特征矩阵 (假设 data.x 存在)
    feat = data.x.cpu().numpy()  # 确保转移到 CPU 并转为 numpy
    np.save(save_path + "_feat.npy", feat)

    # 3. 保存标签
    label = data.y.flatten().cpu().numpy()  # 展平并转为 numpy
    np.save(save_path + "_label.npy", label)
    # ================== 保存结束 ==================
