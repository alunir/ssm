import math
from dataclasses import dataclass
from typing import Union


@dataclass
class ModelArgs:
    d_input: int
    d_model: int
    d_output: int
    n_layer: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    device: str = "cpu"

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
