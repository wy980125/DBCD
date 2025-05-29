import os
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import load_data, args, utils
from model import myGCN
from loss import Q_loss, f_loss
from community import community_louvain
from consensus import (
    detect_louvain_communities, detect_kmeans_communities_data,
    compute_consensus_matrix, build_joint_onehot_from_labels,
    filter_large_communities
)
from load_data2 import uni_load_data


def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


def compute_statistics(seed_results):
    all_metrics = set().union(*[res.keys() for res in seed_results])
    return {
        **{f'avg_{m}': round(float(np.mean([res.get(m, 0) for res in seed_results])), 4) for m in all_metrics},
        **{f'std_{m}': round(float(np.std([res.get(m, 0) for res in seed_results])), 4) for m in all_metrics}
    }


def run_experiment(preprocessed_data, lr, beta, seed):
    utils.set_seed(seed)
    device = args.device
    loss_history = []

    data = preprocessed_data['data']
    G = preprocessed_data['G']
    labels = preprocessed_data['labels']
    torch_sparse_adj = preprocessed_data['torch_sparse_adj']
    degree = preprocessed_data['degree']
    B = preprocessed_data['B']
    sparse_adj = preprocessed_data['sparse_adj']
    num_edges = preprocessed_data['num_edges']
    n_clusters = preprocessed_data['n_clusters']
    filtered_indices = preprocessed_data['filtered_indices']
    onehot_matrix_tensor = preprocessed_data['onehot_matrix_tensor']

    model = myGCN(
        num_features=data.num_features,
        hidden_dim=256,
        num_communities=n_clusters
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = 0.0
    best_results = {}
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        Z = model(data.x.to(device), data.edge_index.to(device))

        q_loss = Q_loss(Z, data.num_nodes, num_edges, sparse_adj, degree, device)
        filtered_embeddings = Z[filtered_indices]
        aux_loss = f_loss(filtered_embeddings, onehot_matrix_tensor)
        total_loss = q_loss + beta * aux_loss

        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                Z_val = model(data.x.to(device), data.edge_index.to(device))
                label_pred = Z_val.argmax(dim=1).cpu().numpy()
                if len(np.unique(label_pred)) > 1:
                    CH_score = utils.CH_score(data.x.cpu().numpy(), label_pred)
                else:
                    CH_score = 0.0

                results = utils.evaluate_louvain(
                    label_pred, labels, preprocessed_data['num_nodes'],
                    num_edges, torch_sparse_adj, degree, G, device
                )
                results['CH_score'] = CH_score
                results['score'] = CH_score * results['modularity']

                if results['score'] > best_score:
                    best_score = results['score']
                    best_results = results

    total_time = time.time() - start_time
    print(f'Seed {seed} training finished in {total_time:.2f}s')
    return best_results, loss_history


def batch_experiment(dataset_names):
    lrs = [0.002]
    betas = [0.6]
    seeds = range(10)


    for dataset in dataset_names:
        print(f'Processing dataset: {dataset}')
        data, edge_index, num_nodes, num_edges, labels, sparse_adj, torch_sparse_adj = uni_load_data(dataset, args.device)
        degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float().to(args.device)
        G = load_data.get_graph(data)
        B, A = utils.compute_modularity_matrix(data)

        comm_A = detect_louvain_communities(G)
        filtered_comm_A = filter_large_communities(comm_A)
        n_clusters = len(set(filtered_comm_A.values()))

        comm_B = detect_kmeans_communities_data(G, n_clusters, data)
        filtered_nodes = set(filtered_comm_A.keys())
        consensus_matrix, node_index = compute_consensus_matrix(filtered_nodes, filtered_comm_A, comm_B)
        filtered_indices = torch.tensor(list(node_index.keys()), dtype=torch.long).to(args.device)

        C_joint, _ = build_joint_onehot_from_labels(filtered_nodes, filtered_comm_A, comm_B)
        C_joint = C_joint.float().to(args.device)

        preprocessed_data = {
            'data': data,
            'datasetname': dataset,
            'G': G,
            'edge_index': edge_index,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'labels': labels,
            'torch_sparse_adj': torch_sparse_adj,
            'degree': degree,
            'B': B,
            'A': A,
            'in_dim': data.x.shape[1],
            'sparse_adj': sparse_adj,
            'filtered_indices': filtered_indices,
            'onehot_matrix_tensor': C_joint,
            'n_clusters': n_clusters
        }

        base_result_dir = Path('result_e2e') / dataset
        base_result_dir.mkdir(parents=True, exist_ok=True)
        excel_path = base_result_dir / f'{dataset}_results_e2e.xlsx'

        try:
            df = pd.read_excel(excel_path)
        except FileNotFoundError:
            df = pd.DataFrame()

        best_hyper = {'lr': None, 'beta': None, 'F1': -1}

        for lr, beta in product(lrs, betas):
            print(f'Running {dataset} with lr={lr}, beta={beta}')
            seed_results = []
            for seed in seeds:
                best_result, loss_history = run_experiment(preprocessed_data, lr, beta, seed)

                seed_results.append(convert_numpy_types(best_result))


            stats = compute_statistics(seed_results)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            result_row = {
                'timestamp': timestamp,
                'lr': lr,
                'epoch': args.epochs,
                'beta': beta,
                **{f"{m.split('_')[1]}_{m.split('_')[0]}": v for m, v in stats.items() if m.startswith(('avg_', 'std_'))}
            }

            df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)

            json_path = base_result_dir / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
            with open(json_path, 'w') as f:
                json.dump({
                    'lr': lr,
                    'beta': beta,
                    'epochs': args.epochs,
                    'statistics': stats,
                    'details': {f'seed_{i}': r for i, r in enumerate(seed_results)}
                }, f, indent=4)

            if stats.get('avg_F1', 0.0) > best_hyper['F1']:
                best_hyper = {
                    'lr': lr,
                    'beta': beta,
                    'epochs': args.epochs,
                    'F1': stats['avg_F1'],
                    'file_path': str(json_path)
                }

        df.to_excel(excel_path, index=False, float_format="%.4f")

        final_dir = Path('batchfinal_results_e2e') / dataset
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / f'best_{datetime.now().strftime("%Y%m%d-%H%M%S")}.json'

        with open(final_path, 'w') as f:
            json.dump(convert_numpy_types({
                'dataset': dataset,
                'best_hyperparameters': best_hyper
            }), f, indent=4)

        print(f"Finished processing {dataset}")


if __name__ == '__main__':
    args = args.parse_args()
    dataset_name = ['cora']
    batch_experiment(dataset_name)
