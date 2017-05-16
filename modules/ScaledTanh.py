import torch.nn as nn

class ScaledTanh(nn.Module):
    def __init__(self, min, max):
        super(ScaledTanh, self).__init__()
        self.tanh = nn.Tanh()
        self.scale = (max - min) / 2
        self.shift = self.scale + min

    def forward(self, input):
        tanh = self.tanh.forward(input)
        return self.scale * tanh + self.shift