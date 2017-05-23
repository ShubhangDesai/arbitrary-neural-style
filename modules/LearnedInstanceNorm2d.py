from torch.nn import InstanceNorm2d

class LearnedInstanceNorm2d(InstanceNorm2d):
    def __init__(self, num_features, weight, bias, eps=1e-5, momentum=0.1):
        super(LearnedInstanceNorm2d, self).__init__(num_features, eps, momentum, affine=True)

        self.weight = weight
        self.bias = bias
