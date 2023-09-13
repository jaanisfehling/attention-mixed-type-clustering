from scipy.optimize import linear_sum_assignment
import torch
import numpy as np


def cluster_accuracy(labels_true, labels_pred):
    # We need to map the labels to our cluster labels
    # This is a linear assignment problem on a bipartite graph
    k = max(len(np.unique(labels_true)), len(np.unique(labels_pred)))
    cost_matrix = np.zeros((k, k))
    for i in range(labels_true.shape[0]):
        cost_matrix[labels_true[i], labels_pred[i]] += 1
    inverted_cost_matrix = cost_matrix.max() - cost_matrix
    row_ind, col_ind = linear_sum_assignment(inverted_cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / labels_pred.size

def build_block(layers: list, activation_fn: torch.nn.Module = torch.nn.LeakyReLU, output_fn: torch.nn.Module = torch.nn.LeakyReLU, 
                bias: bool = True, batch_norm: bool = False, dropout: float = None):
    block_list = []
    for i in range(len(layers) - 1):
        block_list.append(torch.nn.Linear(layers[i], layers[i + 1], bias=bias))
        if batch_norm:
            block_list.append(torch.nn.BatchNorm1d(layers[i + 1]))
        if dropout is not None:
            block_list.append(torch.nn.Dropout(dropout))
        if activation_fn is not None:
            if (i != len(layers) - 2):
                block_list.append(activation_fn())
            else:
                if output_fn is not None:
                    block_list.append(output_fn())
    return torch.nn.Sequential(*block_list)

def build_autoencoder(input_dim: int, output_dim: int, layer_per_block: int, hidden_dim: int = None, activation_fn: torch.nn.Module = torch.nn.LeakyReLU, 
                      output_fn: torch.nn.Module = torch.nn.LeakyReLU, bias: bool = True, batch_norm: bool = False, dropout: float = None):
    if not hidden_dim:
        hidden_dim = max(1, min(round(input_dim/4), round(output_dim/4)))

    encoder_layer_list = list(range(input_dim, hidden_dim - 1, min(-1, -round((input_dim - hidden_dim) / layer_per_block))))
    encoder_layer_list[-1] = hidden_dim
    encoder = build_block(encoder_layer_list, activation_fn, output_fn, bias, batch_norm, dropout)

    decoder_layer_list = list(range(output_dim, hidden_dim - 1, min(-1, -round((output_dim - hidden_dim) / layer_per_block))))
    decoder_layer_list[-1] = hidden_dim
    decoder = build_block(decoder_layer_list[::-1], activation_fn, output_fn, bias, batch_norm, dropout)

    return encoder, decoder
