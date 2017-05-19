import torch
import torch.nn as nn

class GramMatrix(nn.Module):
    def forward(self, input):
        N, C, H, W = input.size()  # a=batch size(=1)
        features = input.view(N, C, H * W)
        G = torch.bmm(features, features.permute(0, 2, 1))

        return G.div(C * H * W)