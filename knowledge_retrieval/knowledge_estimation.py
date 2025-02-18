# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .gnn_predictor import GNNPredictor, pretrain_gnn_predictor
from collections import defaultdict


design_dimensions = {
    'node_classification': ['neigh', 'norm', 'agg', 'comb', 'l_mp', 'stage'],
    'link_prediction': ['decode', 'norm', 'agg', 'comb', 'l_mp', 'stage'],
    'graph_classification': ['decode', 'norm', 'agg', 'comb', 'l_mp', 'stage']
}


class KnowledgeEstimation:
    def __init__(self, task, buffer_size=10):
        self.task = task
        if task == 'node_classification':
            self.base_db_path = "knowledge_retrieval/knowledge_base/node"
        elif task == 'link_prediction':
            self.base_db_path = "knowledge_retrieval/knowledge_base/link"
        elif task == 'graph_classification':
            self.base_db_path = "knowledge_retrieval/knowledge_base/graph"
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.estimators = {}
        self.optimizers = {}
        self.feedback_buffer = []
        self.buffer_size = buffer_size
        self.data_cache = {}

    def initialize_estimator(self, benchmark_datasets):
        for dataset_name in benchmark_datasets:
            print(f"Initialize estimator for {dataset_name}")
            data_obj = self.prepare_data(dataset_name)
            model = self.load_pretrained_model(dataset_name, data_obj)
            model.to(self.device)
            self.estimators[dataset_name] = model
            self.optimizers[dataset_name] = optim.Adam(model.parameters(), lr=1e-3)
            self.data_cache[dataset_name] = data_obj
    
    def add_feedback_observation(self, current_model, modified_model, unseen_gain, similarity_dict):
        """
        Add a new observation to the feedback buffer.

        Parameters:
        - benchmark_dataset_name: str
        - current_model: dict representing current architecture
        - modified_model: dict representing modified architecture
        - benchmark_gain: float
        - unseen_gain: float
        """
        edge_dict = {}  # {ds_name -> (src_node, dst_node)}

        # For each dataset that might appear in similarity_dict:
        # (or for each ds_name in self.estimators if you prefer)
        for ds_name in similarity_dict:
            sim = similarity_dict[ds_name]
            if sim == 0.0:
                # If similarity = 0, this GNN won't contribute to the final pred,
                # so we can skip arch2node entirely.
                continue

            data_obj = self.data_cache.get(ds_name, None)
            if data_obj is None:
                data_obj = self.prepare_data(ds_name)
                self.data_cache[ds_name] = data_obj

            arch_cols = data_obj.arch_cols
            arch2node = data_obj.arch2node

            # Convert current_model -> tuple
            curr_tuple = tuple(
                int(current_model[col]) if str(current_model[col]).isdigit() else str(current_model[col])
                for col in arch_cols
            )
            # Convert modified_model -> tuple
            mod_tuple = tuple(
                int(modified_model[col]) if str(modified_model[col]).isdigit() else str(modified_model[col])
                for col in arch_cols
            )

            # If either arch is missing, skip
            if curr_tuple not in arch2node or mod_tuple not in arch2node:
                continue

            src_node = arch2node[curr_tuple]
            dst_node = arch2node[mod_tuple]
            edge_dict[ds_name] = (src_node, dst_node)

        # Save in the buffer
        self.feedback_buffer.append((edge_dict, unseen_gain, similarity_dict))

        # Maintain buffer size
        if len(self.feedback_buffer) > self.buffer_size:
            self.feedback_buffer.pop(0)

    def feedback_integration(self, finetune_datasets, replay_ratio=0):
        """
        Efficient approach:
          - For each dataset, gather edges from the entire buffer into one big batch
          - Predict them in a single forward pass
          - Then combine predictions across all datasets for each observation
          - Compute final L1 loss, backprop once
        """
        if not self.feedback_buffer:
            print("[FEEDBACK] Buffer is empty. Nothing to integrate.")
            return
        
        # If user hasn't specified which datasets to fine-tune, default to all
        if finetune_datasets is None:
            finetune_datasets = list(self.estimators.keys())
        else:
            # Only keep those that exist in self.estimators
            finetune_datasets = [ds for ds in finetune_datasets if ds in self.estimators]

        if not finetune_datasets:
            print("[FEEDBACK] No valid finetune datasets specified. Skipping integration.")
            return
        
        # Put each chosen GNN into train mode
        for ds_name in finetune_datasets:
            self.estimators[ds_name].train()

        loss_fn = nn.L1Loss()

        # We'll store edges per dataset from the buffer, plus the direct unseen gains.
        edges_by_ds = defaultdict(list)         # ds_name -> list[(src, dst)]
        gains_by_ds = defaultdict(list)         # ds_name -> list[float], matching edges_by_ds

        # 1) Gather edges from the feedback buffer
        #    Each entry in self.feedback_buffer is (edge_dict, unseen_gain, sim_dict)
        #    ignoring sim_dict entirely
        for (edge_dict, unseen_gain, _sim_dict) in self.feedback_buffer:
            # For each dataset that is in finetune_datasets
            for ds_name in finetune_datasets:
                if ds_name in edge_dict:
                    (src_node, dst_node) = edge_dict[ds_name]
                    edges_by_ds[ds_name].append((src_node, dst_node))
                    # Use the direct unseen_gain as the target
                    gains_by_ds[ds_name].append(unseen_gain)

        # 2) For each dataset, do a single forward pass, compute L1Loss vs unseen_gain
        for ds_name in finetune_datasets:
            ds_edges = edges_by_ds[ds_name]        # Buffer edges
            ds_unseen_gains = gains_by_ds[ds_name] # Buffer direct gains

            gnn_model = self.estimators[ds_name]
            optimizer = self.optimizers[ds_name]
            data_obj = self.data_cache[ds_name].to(self.device)

            if not ds_edges:
                # No observations referencing this dataset in the buffer => skip
                continue

            # =========== Step A: Build the buffer modifications set ===========
            buffer_modifications = set(ds_edges)  # set of (src, dst)

            # =========== Step B: Replay Logic ===========
            # We'll pick `replay_ratio * len(ds_edges)` edges from data_obj.edge_index
            # that do not appear in buffer_modifications. We'll treat data_obj.edge_improvements
            # as the "gain" for those edges, ignoring any similarity logic.
            num_buffer = len(ds_edges)
            num_replay = num_buffer * replay_ratio

            replay_edges = []
            replay_gains = []

            if num_replay > 0:
                all_edge_list = data_obj.edge_index.t().tolist()   # shape => [[src, dst], ...]
                all_gain_list = data_obj.edge_improvements.tolist()

                # Filter out buffer edges
                available_edges = []
                available_gains = []
                for e, g in zip(all_edge_list, all_gain_list):
                    e_tuple = tuple(e)
                    if e_tuple not in buffer_modifications:
                        available_edges.append(e)
                        available_gains.append(g)

                if len(available_edges) < num_replay:
                    print(f"[REPLAY] {ds_name}: Not enough available edges for replay. Using {len(available_edges)} instead of {num_replay}.")
                    num_replay = len(available_edges)

                if num_replay > 0:
                    selected_indices = np.random.choice(len(available_edges), size=num_replay, replace=False)
                    for idx in selected_indices:
                        e = available_edges[idx]
                        g = available_gains[idx]
                        # We'll treat g as the direct gain
                        replay_edges.append((e[0], e[1]))
                        replay_gains.append(g)

            # Combine buffer + replay
            combined_edges = ds_edges + replay_edges
            combined_targets = ds_unseen_gains + replay_gains

            if not combined_edges:
                print(f"[FEEDBACK] No samples found for {ds_name} after replay. Skipping.")
                continue

            # =========== Step C: Single forward pass with combined edges ===========
            edge_tensor = torch.tensor(combined_edges, dtype=torch.long, device=self.device).t()  # shape => [2, num_samples]
            target_tensor = torch.tensor(combined_targets, dtype=torch.float, device=self.device)  # shape => [num_samples]

            node_emb = gnn_model(data_obj)
            ds_pred = gnn_model.predict_edges(node_emb, edge_tensor)  # shape => [num_samples, 1]
            ds_pred = ds_pred.squeeze(dim=1)  # shape => [num_samples]

            # =========== Step D: Compute L1Loss ignoring similarity ===========
            ds_loss = loss_fn(ds_pred, target_tensor)

            # =========== Step E: Backprop & update for this dataset's optimizer ===========
            optimizer.zero_grad()
            ds_loss.backward()
            optimizer.step()

    def prepare_data(self, dataset_name):
        """
        Prepare and load the data, constructing the model-model graph if needed.
        If processed data already exists, load it. Otherwise, create it.
        """
        dataset_path = self.get_dataset_path(dataset_name)
        graph_pt = os.path.join(dataset_path, "model_graph.pt")
        if os.path.exists(graph_pt):
            # Load processed data
            data_obj = torch.load(graph_pt)
            return data_obj
        else:
            # Read the CSV and construct the graph
            csv_path = self.get_csv_path(dataset_name)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found at {csv_path}")

            df = pd.read_csv(csv_path, sep=',')  # Adjust delimiter if needed
            arch_cols = design_dimensions[self.task]
            
            # Identify unique architectures
            arch_df = df[arch_cols].drop_duplicates()
            arch_df['arch_id'] = np.arange(len(arch_df))
            df = df.merge(arch_df, on=arch_cols, how='left')

            # Aggregate performance metric (assume 'accuracy' as example)
            perf_metric = 'accuracy'
            arch_perf = df.groupby('arch_id')[perf_metric].mean().to_dict()

            # Encode node features
            node_feats_list = []
            for c in arch_cols:
                le = LabelEncoder()
                arr = le.fit_transform(arch_df[c])
                node_feats_list.append(arr.reshape(-1,1))
            node_feats = np.hstack(node_feats_list)
            node_feats = torch.tensor(node_feats, dtype=torch.float)
            arch_list = arch_df.to_dict('records')

            edges = []
            edge_improvements = []
            for i, archA in enumerate(arch_list):
                a_id = archA['arch_id']
                perfA = arch_perf[a_id]
                for j, archB in enumerate(arch_list):
                    if i == j:
                        continue
                    if self.is_one_hop(archA, archB, arch_cols):
                        b_id = archB['arch_id']
                        perfB = arch_perf[b_id]
                        improvement = perfB - perfA
                        edges.append((a_id, b_id))
                        edge_improvements.append(improvement)

            edges = np.array(edges)
            edge_improvements = torch.tensor(edge_improvements, dtype=torch.float)

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data_obj = Data(x=node_feats, edge_index=edge_index)
            data_obj.edge_improvements = edge_improvements

            # Create arch2node mapping
            arch2node = {}
            for _, row in arch_df.iterrows():
                arch_tuple = tuple(row[c] for c in arch_cols)
                arch2node[arch_tuple] = row['arch_id']
            
            # Attach arch2node to data_obj
            # data_obj is a PyG Data object which can store arbitrary attributes
            data_obj.arch2node = arch2node
            data_obj.arch_cols = arch_cols  # Store the order of arch_cols for reference

            # Save processed data
            torch.save(data_obj, graph_pt)
            return data_obj

    @staticmethod
    def is_one_hop(archA, archB, arch_cols):
        diff_count = sum((archA[c] != archB[c]) for c in arch_cols)
        return diff_count == 1

    def get_dataset_path(self, dataset_name):
        return os.path.join(self.base_db_path, dataset_name.lower())

    def get_csv_path(self, dataset_name):
        # Example: knowledge_retrieval/node/cora/agg/val_best.csv
        # This path structure can be adjusted as needed.
        dataset_path = self.get_dataset_path(dataset_name)
        csv_path = os.path.join(dataset_path, 'agg', 'val_best.csv')
        return csv_path

    def get_pretrained_model_path(self, dataset_name, temp=False):
        """
        Return path for pretrained model. If not found, we will train and save it.
        """
        dataset_path = self.get_dataset_path(dataset_name)
        model_path = os.path.join(dataset_path, "ecc_predictor.pt" if not temp else "temp_ecc_predictor.pt")
        return model_path

    def load_pretrained_model(self, dataset_name, data_obj):
        """
        Load or create a pretrained GNNPredictor model for the given dataset.
        """
        model_path = self.get_pretrained_model_path(dataset_name)
        temp_model_path = self.get_pretrained_model_path(dataset_name, temp=True)
        if os.path.exists(model_path):
            # Load pretrained model
            best_model = torch.load(model_path, map_location=self.device)
        else:
            best_emd = float('inf')
            best_model = None

            num_repeats = 10
            for repeat in range(1, num_repeats + 1):
                # Use a unique model path for each repeat to avoid overwriting
                model, emd = pretrain_gnn_predictor(
                    data_obj, self.device, temp_model_path
                )
                print(f"Iteration {repeat}: EMD: {emd}")
                
                if emd < best_emd:
                    best_emd = emd
                    best_model = model
                    # Save the best model
                    torch.save(best_model, model_path)
                    print(f"New best model found. Saved to {model_path}.")
                
                # Remove the temporary model file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
            
            print(f"\nBest model has EMD: {best_emd}")
        return best_model

    def create_in_process_model(self, base_model):
        """
        Create an in-process copy of the model for feedback integration, without modifying the pretrained model.
        """
        in_process_model = GNNPredictor(in_channels=base_model.conv1.in_channels,
                                        hidden_channels=base_model.conv1.out_channels,
                                        out_channels=1).to(self.device)
        # Copy parameters from base_model to in_process_model
        in_process_model.load_state_dict(base_model.state_dict())
        return in_process_model

    def remove_in_process_model(self, in_process_model):
        """
        Remove or cleanup the in-process model after finishing refinement.
        Just dereference it here, GC will handle the rest.
        """
        del in_process_model

