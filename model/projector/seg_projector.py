import torch
import torch.nn as nn

class SegProjector(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer that maps input to output of the same size
        self.linear = nn.Linear(4096, 4096)
        # Define the GeLU activation function
        self.gelu = nn.GELU()

    def forward(self, x):
        # Apply the linear layer
        x = self.linear(x)
        # Apply the GeLU activation function
        x = self.gelu(x)
        return x

def seg_projector_builder():
    return SegProjector()
