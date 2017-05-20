import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        N, C, H, W = input.size()
        output = input.view(N * C, H * W)
        return output
