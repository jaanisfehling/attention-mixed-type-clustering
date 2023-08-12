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


class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, head_dim=8):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = head_dim * heads
        
        self.to_keys = nn.Linear(dim, inner_dim, bias=False)
        self.to_queries = nn.Linear(dim, inner_dim, bias=False)
        self.to_values = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # => x = b x d

        q = self.to_queries(x)
        k = self.to_keys(x)
        v = self.to_values(x)
        # => q/k/v = b x (h h_d)

        b = q.size()[0]
        q = q.view(b, self.heads, self.head_dim)
        k = k.view(b, self.heads, self.head_dim)
        v = v.view(b, self.heads, self.head_dim)
        # => q/k/v = b x h x h_d

        q = q.view(b * self.heads, self.head_dim)
        k = k.view(b * self.heads, self.head_dim)
        v = v.view(b * self.heads, self.head_dim)
        # => q/k/v = (b h) x h_d

        x = scaled_dot_product_attention(q, k, v)

        x = x.view(b, self.heads, self.head_dim)
        # => x = b x h x h_d

        x = x.view(b, self.heads * self.head_dim)
        # => x = b x (h h_d)

        x = self.to_out(x)
        return x
       

class Transformer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attention = Attention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fed_forward = self.feed_forward(x)
        x = self.norm2(fed_forward + x)
        return x


class TransformerAutoencoder(torch.nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, input_dim: int, cat_dim: int,
                 embedding_sizes: List[Tuple[int, int]], depth: int = 8, device: torch.device | str = "cpu"):
        super().__init__()
        self.fitted = False

        self.embeddings = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_sizes])
        transformer_list = [Transformer(cat_dim) for _ in range(depth)]
        self.transformers = nn.Sequential(*transformer_list)
        self.encoder = encoder
        self.decoder = decoder

        self.device = device
        self.to(self.device)

    def encode(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], 1)
        x_cat = x_cat.to(torch.float)
        self.last_target = torch.cat((x_cat, x_cont), 1).clone().detach()

        x_cat = self.transformers(x_cat)

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
