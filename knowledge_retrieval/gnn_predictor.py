# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import EdgeConv
import numpy as np

from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics import r2_score


class GNNPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNPredictor, self).__init__()

        self.conv1 = EdgeConv(nn=nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = EdgeConv(nn=nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))

        # Edge-level predictor: We'll use the final node embeddings from GNN
        # and combine them to predict edge improvements.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x  # node embeddings

    def predict_edges(self, node_emb, edge_index):
        # node_emb shape: [num_nodes, hidden_channels]
        src = node_emb[edge_index[0]]
        dst = node_emb[edge_index[1]]
        edge_input = torch.cat([src, dst - src], dim=-1)
        return self.mlp(edge_input)
    
    def predict_benchmark_gains(self, data_obj, current_model, modifications, device):
        """
        Predict the benchmark gains for multiple modifications at once.

        Parameters:
        - data_obj: Data object with arch2node and arch_cols.
        - current_model: dict for current architecture.
        - modifications: list of (modified_model, new_benchmark_accuracy, _) tuples.
        - device: torch device.

        Returns:
        - predicted_gains: list of predicted improvements for each modification in the same order.
        """
        arch_cols = data_obj.arch_cols
        arch2node = data_obj.arch2node

        # Convert current_model to tuple
        current_arch_tuple = tuple(
            int(current_model[col]) if str(current_model[col]).isdigit() else str(current_model[col])
            for col in arch_cols
        )

        if current_arch_tuple not in arch2node:
            raise ValueError(f"Current architecture {current_arch_tuple} not found in arch2node mapping.")

        src_node = arch2node[current_arch_tuple]

        # Build arrays for edges
        dst_nodes = []
        for (modified_model, _, _) in modifications:
            modified_arch_tuple = tuple(
                int(modified_model[col]) if str(modified_model[col]).isdigit() else str(modified_model[col])
                for col in arch_cols
            )
            if modified_arch_tuple not in arch2node:
                raise ValueError(f"Modified architecture {modified_arch_tuple} not found in arch2node mapping.")
            dst_node = arch2node[modified_arch_tuple]
            dst_nodes.append(dst_node)

        src_nodes = [src_node] * len(dst_nodes)
        edge_idx = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)

        self.eval()
        data_obj = data_obj.to(device)
        with torch.no_grad():
            node_emb = self(data_obj)
            pred = self.predict_edges(node_emb, edge_idx)  # shape: [num_modifications, 1]
            predicted_gains = pred.squeeze().cpu().tolist()

        return predicted_gains


def pretrain_gnn_predictor(data_obj, device, model_path, val_ratio=0.05, max_epochs=5000):
    """
    Pretrain the GNN predictor with a fixed training/validation split.
    By default, uses a 9.5:0.5 ratio (95% training, 5% validation).
    """
    model = GNNPredictor(in_channels=data_obj.x.size(1), hidden_channels=64, out_channels=1).to(device)
    data_obj = data_obj.to(device)
    edge_improvements = data_obj.edge_improvements.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.L1Loss()

    # Prepare fixed train/val split
    num_edges = data_obj.edge_index.size(1)
    val_size = math.ceil(val_ratio * num_edges)
    torch.manual_seed(42)
    indices = torch.randperm(num_edges)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_edge_index = data_obj.edge_index[:, train_indices]
    train_edge_improvements = edge_improvements[train_indices]

    val_edge_index = data_obj.edge_index[:, val_indices]
    val_edge_improvements = edge_improvements[val_indices]

    best_val_loss = float('inf')
    patience = 20
    no_improve_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        node_emb = model(data_obj)
        pred_train = model.predict_edges(node_emb, train_edge_index)
        train_loss = loss_fn(pred_train.squeeze(), train_edge_improvements)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            node_emb = model(data_obj)
            pred_val = model.predict_edges(node_emb, val_edge_index)
            val_loss = loss_fn(pred_val.squeeze(), val_edge_improvements)

        print(f"Epoch {epoch+1:02d}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        # Early stopping based on validation loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            no_improve_epochs = 0
            torch.save(model, model_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("No improvement on validation set for several epochs, stopping early.")
                print(f"Loss {best_val_loss} - {epoch - patience + 1} epochs trained.")
                break
        
    model = torch.load(model_path, map_location=device)
    _, _, _, _, _, emd = evaluate_model_on_all_edges(model, data_obj, device)

    return model, emd

def evaluate_model_on_all_edges(model, data_obj, device):
    """
    Evaluate the model on all edges present in data_obj, computing MSE, correlation (Pearson),
    and R² score.

    This evaluates how well the model fits the entire distribution. Note that if the model was
    trained on these same edges, this does not measure generalization but rather how well the model 
    has captured the training distribution.

    model: trained GNNPredictor
    data_obj: Data object containing node features, edge_index, and edge_improvements
    device: torch device
    """
    model.eval()
    data_obj = data_obj.to(device)
    edge_improvements = data_obj.edge_improvements.to(device)

    with torch.no_grad():
        node_emb = model(data_obj)
        pred_all = model.predict_edges(node_emb, data_obj.edge_index.to(device))
        pred_all = pred_all.squeeze().cpu().numpy()
        true_all = edge_improvements.cpu().numpy()

    # Compute MSE
    mse_loss = ((pred_all - true_all)**2).mean()
    mae_loss = np.abs(pred_all - true_all).mean()
    correlation, _ = pearsonr(pred_all, true_all)
    r2 = r2_score(true_all, pred_all)
    sign_consistency = ((pred_all * true_all) > 0).mean()
    emd = wasserstein_distance(true_all, pred_all)

    print(f"All Edges MSE: {mse_loss:.4f}")
    print(f"All Edges MAE: {mae_loss:.4f}")
    print(f"All Edges Pearson Correlation: {correlation:.4f}")
    print(f"All Edges R² Score: {r2:.4f}")
    print(f"All Edges Sign Consistency: {sign_consistency:.4f}")
    print(f"All Edges EMD: {emd:.4f}\n")

    return mse_loss, correlation, r2, mae_loss, sign_consistency, emd

