import math
from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    attn_weights = torch.matmul(q, k.transpose(-2, -1))
    vector_dim = q.size()[-1]
    attn_weights = attn_weights / math.sqrt(vector_dim)
    attention = F.softmax(attn_weights, dim=-1)
    values = torch.matmul(attention, v)
    return values


class CatToCatAutoencoder(torch.nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, input_dim: int, cat_dim: int,
                 embedding_sizes: List[Tuple[int, int]], attention: bool = False, device: torch.device | str = "cpu"):
        super().__init__()
        self.fitted = False

        self.attention = attention
        if self.attention:
            self.to_keys = nn.Linear(cat_dim, cat_dim, bias=False)
            self.to_queries = nn.Linear(cat_dim, cat_dim, bias=False)
            self.to_values = nn.Linear(cat_dim, cat_dim, bias=False)

        self.encoder = encoder
        self.decoder = decoder
        self.embeddings = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_sizes])

        self.device = device
        self.to(self.device)

    def encode(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], 1)
        x_cat = x_cat.to(torch.float)
        self.last_target = x_cat.clone().detach()

        if self.attention:
            q = self.to_queries(x_cat)
            k = self.to_keys(x_cat)
            v = self.to_values(x_cat)
            x_cat = scaled_dot_product_attention(q, k, v)

        x = self.encoder(x_cat)
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

