"""
@authors:
Lukas Miklautz
Dominik Mautz
"""

from ._utils_duped import detect_device, encode_batchwise, \
    squared_euclidean_distance, predict_batchwise
from clustpy.deep._train_utils import get_trained_autoencoder
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _dcn(dataloader: torch.utils.data.DataLoader, n_clusters: int, batch_size: int, pretrain_learning_rate: float,
         clustering_learning_rate: float, pretrain_epochs: int,
         clustering_epochs: int, optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
         autoencoder: torch.nn.Module, embedding_size: int, degree_of_space_distortion: float,
         degree_of_space_preservation: float, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    device = detect_device()
    trainloader = dataloader
    testloader = dataloader
    autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                          optimizer_class, loss_fn, 0, embedding_size, autoencoder)
    # Execute kmeans in embedded space
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    init_centers = kmeans.cluster_centers_
    # Setup DCN Module
    dcn_module = _DCN_Module(init_centers).to_device(device)
    # Use DCN learning_rate (usually pretrain_learning_rate reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()), lr=clustering_learning_rate)
    # DEC Training loop
    dcn_module.fit(autoencoder, trainloader, clustering_epochs, device, optimizer, loss_fn,
                   degree_of_space_distortion, degree_of_space_preservation)
    # Get labels
    dcn_labels = predict_batchwise(testloader, autoencoder, dcn_module, device)
    dcn_centers = dcn_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dcn_labels, dcn_centers, autoencoder


def _compute_centroids(centers: torch.Tensor, embedded: torch.Tensor, count: torch.Tensor, labels: torch.Tensor) -> (
        torch.Tensor, torch.Tensor):
    for i in range(embedded.shape[0]):
        c = labels[i].item()
        count[c] += 1
        eta = 1.0 / count[c].item()
        centers[c] = (1 - eta) * centers[c] + eta * embedded[i]
    return centers, count


class _DCN_Module(torch.nn.Module):
    def __init__(self, init_np_centers: np.ndarray):
        super().__init__()
        self.centers = torch.tensor(init_np_centers)

    def dcn_loss(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        dist = squared_euclidean_distance(embedded, self.centers, weights=weights)
        loss = (dist.min(dim=1)[0]).mean()
        return loss

    def predict_hard(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        dist = squared_euclidean_distance(embedded, self.centers, weights=weights)
        labels = (dist.min(dim=1)[1])
        return labels

    def update_centroids(self, embedded: torch.Tensor, count: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.centers, count = _compute_centroids(self.centers, embedded, count, labels)
        return count

    def to_device(self, device: torch.device) -> '_DCN_Module':
        self.centers = self.centers.to(device)
        self.to(device)
        return self

    def fit(self, autoencoder: torch.nn.Module, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
            degree_of_space_distortion: float, degree_of_space_preservation: float) -> '_DCN_Module':
        # DCN training loop
        # Init for count from original DCN code (not reported in Paper)
        # This means centroid learning rate at the beginning is scaled by a hundred
        count = torch.ones(self.centers.shape[0], dtype=torch.int32) * 100
        for _ in range(n_epochs):
            # Update Network
            for batch in trainloader:
                x_cat, x_cont = batch[0].to(device), batch[1].to(device)
                embedded = autoencoder.encode(x_cat, x_cont)
                reconstruction = autoencoder.decode(embedded)
                embedded = torch.cat((embedded, x_cont), 1)
                # compute reconstruction loss
                ae_loss = loss_fn(reconstruction, autoencoder.last_target)
                # compute cluster loss
                cluster_loss = self.dcn_loss(embedded)
                # compute total loss
                loss = degree_of_space_preservation * ae_loss + 0.5 * degree_of_space_distortion * cluster_loss
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update Assignments and Centroids
            with torch.no_grad():
                for batch in trainloader:
                    x_cat, x_cont = batch[0].to(device), batch[1].to(device)
                    embedded = autoencoder.encode(x_cat, x_cont)
                    embedded = torch.cat((embedded, x_cont), 1)

                    ## update centroids [on gpu] About 40 seconds for 1000 iterations
                    ## No overhead from loading between gpu and cpu
                    # count = cluster_module.update_centroid(embedded, count, s)

                    # update centroids [on cpu] About 30 Seconds for 1000 iterations
                    # with additional overhead from loading between gpu and cpu
                    embedded = embedded.cpu()
                    self.centers = self.centers.cpu()

                    # update assignments
                    labels = self.predict_hard(embedded)

                    # update centroids
                    count = self.update_centroids(embedded, count.cpu(), labels.cpu())
                    # count = count.to(device)
                    self.centers = self.centers.to(device)
        return self


class DCNDuped(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters: int, batch_size: int = 256, pretrain_learning_rate: float = 1e-3,
                 clustering_learning_rate: float = 1e-4, pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), degree_of_space_distortion: float = 0.05,
                 degree_of_space_preservation: float = 1.0, autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, random_state: np.random.RandomState = None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_learning_rate = pretrain_learning_rate
        self.clustering_learning_rate = clustering_learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.degree_of_space_distortion = degree_of_space_distortion
        self.degree_of_space_preservation = degree_of_space_preservation
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.random_state = check_random_state(random_state)
        torch.manual_seed(self.random_state.get_state()[1][0])

    def fit(self, dataloader: torch.utils.data.DataLoader) -> 'DCN':
        kmeans_labels, kmeans_centers, dcn_labels, dcn_centers, autoencoder = _dcn(dataloader, self.n_clusters, self.batch_size,
                                                                                   self.pretrain_learning_rate,
                                                                                   self.clustering_learning_rate,
                                                                                   self.pretrain_epochs,
                                                                                   self.clustering_epochs,
                                                                                   self.optimizer_class, self.loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.degree_of_space_distortion,
                                                                                   self.degree_of_space_preservation,
                                                                                   self.random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dcn_labels_ = dcn_labels
        self.dcn_cluster_centers_ = dcn_centers
        self.autoencoder = autoencoder
        return self
