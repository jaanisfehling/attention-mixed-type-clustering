"""
@authors:
Lukas Miklautz
Dominik Mautz
Collin Leiber
"""

from ._utils_duped import detect_device, encode_batchwise, squared_euclidean_distance, predict_batchwise
from clustpy.deep._train_utils import get_trained_autoencoder
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _dec(dataloader: torch.utils.data.DataLoader, n_clusters: int, alpha: float, batch_size: int, pretrain_learning_rate: float,
         clustering_learning_rate: float, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
         autoencoder: torch.nn.Module, embedding_size: int, use_reconstruction_loss: bool,
         cluster_loss_weight: float, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    device = detect_device()
    trainloader = dataloader
    testloader = dataloader
    autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                          optimizer_class, loss_fn, 0, embedding_size, autoencoder)

    # Execute kmeans in embedded space - initial clustering
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    init_centers = kmeans.cluster_centers_
    # Setup DEC Module
    dec_module = _DEC_Module(init_centers, alpha).to(device)
    # Use DEC learning_rate (usually pretrain_learning_rate reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()) + list(dec_module.parameters()),
                                lr=clustering_learning_rate)
    # DEC Training loop
    dec_module.fit(autoencoder, trainloader, clustering_epochs, device, optimizer, loss_fn,
                   use_reconstruction_loss, cluster_loss_weight)
    # Get labels
    dec_labels = predict_batchwise(testloader, autoencoder, dec_module, device)
    dec_centers = dec_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dec_labels, dec_centers, autoencoder


def _dec_predict(centers: torch.Tensor, embedded: torch.Tensor, alpha: float, weights) -> torch.Tensor:
    squared_diffs = squared_euclidean_distance(embedded, centers, weights)
    numerator = (1.0 + squared_diffs / alpha).pow(-1.0 * (alpha + 1.0) / 2.0)
    denominator = numerator.sum(1)
    prob = numerator / denominator.unsqueeze(1)
    return prob


def _dec_compression_value(pred_labels: torch.Tensor) -> torch.Tensor:
    soft_freq = pred_labels.sum(0)
    squared_pred = pred_labels.pow(2)
    normalized_squares = squared_pred / soft_freq.unsqueeze(0)
    sum_normalized_squares = normalized_squares.sum(1)
    p = normalized_squares / sum_normalized_squares.unsqueeze(1)
    return p


def _dec_compression_loss_fn(pred_labels: torch.Tensor) -> torch.Tensor:
    p = _dec_compression_value(pred_labels).detach().data
    loss = -1.0 * torch.mean(torch.sum(p * torch.log(pred_labels + 1e-8), dim=1))
    return loss


class _DEC_Module(torch.nn.Module):
    def __init__(self, init_centers: np.ndarray, alpha: float):
        super().__init__()
        self.alpha = alpha
        # Centers are learnable parameters
        self.centers = torch.nn.Parameter(torch.tensor(init_centers), requires_grad=True)

    def predict(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        pred = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        return pred

    def predict_hard(self, embedded: torch.Tensor, weights=None) -> torch.Tensor:
        pred_hard = self.predict(embedded, weights=weights).argmax(1)
        return pred_hard

    def dec_loss(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        prediction = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        loss = _dec_compression_loss_fn(prediction)
        return loss

    def fit(self, autoencoder: torch.nn.Module, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
            use_reconstruction_loss: bool, cluster_loss_weight: float) -> '_DEC_Module':
        for _ in range(n_epochs):
            for batch in trainloader:
                x_cat, x_cont = batch[0].to(device), batch[1].to(device)
                embedded = autoencoder.encode(x_cat, x_cont)
                cluster_loss = self.dec_loss(torch.cat((embedded, x_cont), 1))
                loss = cluster_loss * cluster_loss_weight
                # Reconstruction loss is not included in DEC
                if use_reconstruction_loss:
                    reconstruction = autoencoder.decode(embedded)
                    ae_loss = loss_fn(autoencoder.last_target, reconstruction)
                    loss += ae_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self


class DECDuped(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters: int, alpha: float = 1.0, batch_size: int = 256, pretrain_learning_rate: float = 1e-3,
                 clustering_learning_rate: float = 1e-4, pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, use_reconstruction_loss: bool = False, cluster_loss_weight: float = 1,
                 random_state: np.random.RandomState = None):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.pretrain_learning_rate = pretrain_learning_rate
        self.clustering_learning_rate = clustering_learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.use_reconstruction_loss = use_reconstruction_loss
        self.cluster_loss_weight = cluster_loss_weight
        self.random_state = check_random_state(random_state)
        torch.manual_seed(self.random_state.get_state()[1][0])

    def fit(self, dataloader: torch.utils.data.DataLoader) -> 'DEC':
        kmeans_labels, kmeans_centers, dec_labels, dec_centers, autoencoder = _dec(dataloader, self.n_clusters, self.alpha,
                                                                                   self.batch_size,
                                                                                   self.pretrain_learning_rate,
                                                                                   self.clustering_learning_rate,
                                                                                   self.pretrain_epochs,
                                                                                   self.clustering_epochs,
                                                                                   self.optimizer_class, self.loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.use_reconstruction_loss,
                                                                                   self.cluster_loss_weight,
                                                                                   self.random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dec_labels_ = dec_labels
        self.dec_cluster_centers_ = dec_centers
        self.autoencoder = autoencoder
        return self


class IDECDuped(DECDuped):
    def __init__(self, n_clusters: int, alpha: float = 1.0, batch_size: int = 256, pretrain_learning_rate: float = 1e-3,
                 clustering_learning_rate: float = 1e-4, pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, random_state: np.random.RandomState = None):
        super().__init__(n_clusters, alpha, batch_size, pretrain_learning_rate, clustering_learning_rate,
                         pretrain_epochs, clustering_epochs, optimizer_class, loss_fn, autoencoder, embedding_size,
                         True, 0.1, random_state)
