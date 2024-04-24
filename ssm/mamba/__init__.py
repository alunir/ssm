import torch.nn as nn

from .sequential import Mamba, ModelArgs


class MambaRegression(Mamba):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.decoder = nn.Sequential(
            nn.Linear(args.d_model, args.d_output), nn.Tanh()  # nn.atanh
        )


class MambaClassifier(Mamba):
    def __init__(self, args: ModelArgs, dim: int = 1):
        super().__init__(args)
        self.decoder = nn.Sequential(
            nn.Linear(args.d_model, args.d_output), nn.LogSoftmax(dim=dim)  # nn.atanh
        )
