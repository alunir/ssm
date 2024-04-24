from typing import List

import torch
import torch.nn as nn

from .lib.s4 import S4Block
from .lib.s4d import S4D


class SingleS4(nn.Module):

    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_layers: int,
        dropout: List[float],
        transposed,
        device="cpu",
        s4d=True,
    ):
        super().__init__()
        self.device = device

        self.encoder = nn.Sequential(nn.Linear(d_input, d_model), nn.GELU())
        if s4d:
            self.layers = [
                S4D(d_model=d_model, dropout=dropout[i], transposed=transposed)
                for i in range(n_layers)
            ]
        else:
            self.layers = [
                S4Block(
                    d_model=d_model,
                    dropout=dropout[i],
                    transposed=transposed,
                    final_act="glu",
                )
                for i in range(n_layers)
            ]

    def setup(self):
        for layer in self.layers:
            layer.to(self.device)
            layer.setup_step()

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x, state = layer(x)
        x = self.decoder(x)
        return x

    @torch.no_grad()
    def step(self, x, states):
        x = self.encoder(x)

        new_states = []
        for layer, state in zip(self.layers, states):
            x, state = layer.step(x, state)
            new_states.append(state)
        x = self.decoder(x)
        return x, new_states

    def get_state(self):
        new_states = []
        for layer in self.layers:
            state = layer.default_state()
            new_states.append(state)

        return new_states


class SingleS4Regression(SingleS4):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        n_layers: int,
        dropout: List[float],
        transposed,
        device="cpu",
        s4d=True,
    ):
        super().__init__(
            d_input=d_input,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            transposed=transposed,
            device=device,
            s4d=s4d,
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_output), nn.Tanh()  # nn.atanh
        )


class SingleS4Classifier(SingleS4):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        n_layers: int,
        dropout: List[float],
        transposed,
        device="cpu",
        s4d=True,
        dim=1,
    ):
        super().__init__(
            d_input=d_input,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            transposed=transposed,
            device=device,
            s4d=s4d,
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_output), nn.LogSoftmax(dim=dim)
        )
