from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class BasicAutoencoder(torch.nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential,
                 embedding_sizes: List[Tuple[int, int]], device: torch.device | str = "cpu"):
        super().__init__()
        self.fitted = False

        self.encoder = encoder
        self.decoder = decoder

        self.device = device
        self.to(self.device)

    def encode(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x_cat = x_cat.to(torch.float)
        self.last_target = torch.cat((x_cat, x_cont), 1)

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
        return loss_fn(reconstruction, torch.cat((x_cat, x_cont), 1))

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
            n_epochs: int = 100, 
            lr: float = 0.001,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
            print_step: int = 25):
        
        optimizer = optimizer_class(params=self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            self.train()
            for batch in dataloader:
                loss = self.loss(batch, loss_fn, self.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if print_step > 0 and ((epoch + 1) % print_step == 0 or epoch == (n_epochs - 1)):
                print(f"Epoch {epoch + 1}/{n_epochs} - Batch Reconstruction loss: {loss.item():.6f}")

        self.fitted = True
        return self

