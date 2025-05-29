import torch
import numpy as np
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, DBLP, NELL
from torch_geometric.utils import to_networkx
import networkx as nx
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


# 1. 数据加载与预处理
def load_data(dataset_name):
    if dataset_name == 'cora':
        dataset = Planetoid(root='data', name="Cora")
    elif dataset_name == 'cite':
        dataset = Planetoid(root='data', name="Citeseer")
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='data', name="PubMed")
    elif dataset_name == 'computers':
        dataset = Amazon(root='data', name='Computers')
    elif dataset_name == 'photo':
        dataset = Amazon(root='data', name='Photo')
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} not implemented.')

    return dataset[0]


def get_graph(data):
    G = to_networkx(data, to_undirected=True)
    return G
