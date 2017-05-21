import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedInstanceNorm2d(nn.Module):

    def __init__(self, num_features, weight, bias, eps=1e-5, momentum=0.1):
        super(LearnedInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = weight
        self.bias = bias
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)

        b, c = input.size(0), input.size(1)

        # Repeat stored stats and affine transform params
        weight = self.weight.repeat(b)
        bias = self.bias.repeat(b)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        input_reshaped = input.contiguous().view(1, b * c, *input.size()[2:])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight, bias,
            self.training, self.momentum, self.eps)

        # Reshape back
        self.running_mean.copy_(running_mean.view(b, c).mean(0))
        self.running_var.copy_(running_var.view(b, c).mean(0))

        return out.view(b, c, *input.size()[2:])