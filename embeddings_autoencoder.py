import math
from typing import List, Tuple
import torch.nn.functional as F

import torch
import numpy as np
from clustpy.deep._early_stopping import EarlyStopping
from torch import nn


def scaled_dot_product_attention(q, k, v):
    attn_weights = torch.matmul(q, k.transpose(-2, -1))
    vector_dim = q.size()[-1]
    attn_weights = attn_weights / math.sqrt(vector_dim)
    attention = F.softmax(attn_weights, dim=-1)
    values = torch.matmul(attention, v)
    return values


class EmbeddingsAutoencoder(torch.nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, input_dim: int, cat_dim: int, embedding_sizes: List[Tuple[int, int]], attention: bool = False):
        super().__init__()
        self.fitted = False

        self.attention = attention
        self.to_keys = nn.Linear(input_dim, input_dim, bias=False)
        self.to_queries = nn.Linear(input_dim, input_dim, bias=False)
        self.to_values = nn.Linear(input_dim, input_dim, bias=False)

        self.encoder = encoder
        self.decoder = decoder
        self.embeddings = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_sizes])

    def encode(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x_cat = x_cat.to(torch.long)
        x = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], 1)
        self.last_target = x.clone().detach()
        x = torch.cat((x, x_cont), 1)

        if self.attention:
            q = self.to_queries(x)
            k = self.to_keys(x)
            v = self.to_values(x)
            x = scaled_dot_product_attention(q, k, v)

        return self.encoder(x)

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

    def fit(self, n_epochs: int, lr: float, batch_size: int = 128, data: np.ndarray = None,
            data_eval: np.ndarray = None,
            dataloader: torch.utils.data.DataLoader = None, evalloader: torch.utils.data.DataLoader = None,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), patience: int = 5,
            scheduler: torch.optim.lr_scheduler = None, scheduler_params: dict = None,
            device: torch.device = torch.device("cpu"), model_path: str = None,
            print_step: int = 0) -> 'EmbeddingsAutoencoder':
        params_dict = {'params': self.parameters(), 'lr': lr}
        optimizer = optimizer_class(**params_dict)

        early_stopping = EarlyStopping(patience=patience)
        if scheduler is not None:
            scheduler = scheduler(optimizer=optimizer, **scheduler_params)
            # Depending on the scheduler type we need a different step function call.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                eval_step_scheduler = True
                if evalloader is None:
                    raise ValueError(
                        "scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, but evalloader is None. Specify evalloader such that validation loss can be computed.")
            else:
                eval_step_scheduler = False
        best_loss = np.inf
        # training loop
        for epoch_i in range(n_epochs):
            self.train()
            for batch in dataloader:
                loss = self.loss(batch, loss_fn, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
            #     print(f"Epoch {epoch_i}/{n_epochs - 1} - Batch Reconstruction loss: {loss.item():.6f}")
            print(f"Epoch {epoch_i}/{n_epochs - 1} - Batch Reconstruction loss: {loss.item():.6f}")

            if scheduler is not None and not eval_step_scheduler:
                scheduler.step()
            # Evaluate autoencoder
            if evalloader is not None:
                # self.evaluate calls self.eval()
                val_loss = self.evaluate(dataloader=evalloader, loss_fn=loss_fn, device=device)
                if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                    print(f"Epoch {epoch_i} EVAL loss total: {val_loss.item():.6f}")
                early_stopping(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch_i
                    # Save best model
                    if model_path is not None:
                        torch.save(self.state_dict(), model_path)

                if early_stopping.early_stop:
                    if print_step > 0:
                        print(f"Stop training at epoch {best_epoch}")
                        print(f"Best Loss: {best_loss:.6f}, Last Loss: {val_loss:.6f}")
                    break
                if scheduler is not None and eval_step_scheduler:
                    scheduler.step(val_loss)
        # Save last version of model
        if evalloader is None and model_path is not None:
            torch.save(self.state_dict(), model_path)
        # Autoencoder is now pretrained
        self.fitted = True
        return self
