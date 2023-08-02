import math
from typing import List, Tuple
import torch.nn.functional as F

import torch
import numpy as np
from clustpy.deep._early_stopping import EarlyStopping
from torch import nn


class BasicAutoencoder(torch.nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, embedding_sizes: List[Tuple[int, int]]):
        super().__init__()
        self.fitted = False

        self.encoder = encoder
        self.decoder = decoder
        self.embeddings = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_sizes])

    def encode(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], 1)
        x_cat = x_cat.to(torch.float)
        self.last_target = torch.cat((x_cat, x_cont), 1).clone().detach()

        x = torch.cat((x_cat, x_cont), 1)
        x = self.encoder(x)
        return x

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x_cat, x_cont))

    def loss(self, batch: list, loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
        x_cat, x_cont = batch[0].to(device), batch[1].to(device)
        reconstruction = self.forward(x_cat, x_cont)
        return loss_fn(reconstruction, self.last_target)

    def evaluate(self, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.modules.loss._Loss, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            loss = 0
            for batch in dataloader:
                loss += self.loss(batch, loss_fn, device)
            loss /= len(dataloader)
        return loss

    def fit(self, 
            dataloader: torch.utils.data.DataLoader,
            n_epochs: int, 
            lr: float,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
            print_step: int = 25) -> 'BasicAutoencoder':
        
        optimizer = optimizer_class(params=self.parameters(), lr=lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(n_epochs):
            self.train()
            for batch in dataloader:
                loss = self.loss(batch, loss_fn, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if print_step > 0 and ((epoch + 1) % print_step == 0 or epoch == (n_epochs - 1)):
                print(f"Epoch {epoch + 1}/{n_epochs} - Batch Reconstruction loss: {loss.item():.6f}")

        self.fitted = True
        return self
