import torch.optim as optim
import torchvision.models as models

from modules.GramMatrix import *
from modules.ScaledTanh import *

class StyleCNN(object):
    def __init__(self, style):
        super(StyleCNN, self).__init__()

        # Initial configurations
        self.style = style
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000
        self.gram = GramMatrix()

        self.use_cuda = torch.cuda.is_available()

        # Build transform network pieces
        self.transform_network, transform_parameters = [], []
        self.out_dims = [32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]
        in_dim, idx = 3, 1
        for out_dim in self.out_dims:
            seq = nn.Sequential() if in_dim != 3 else nn.Sequential(nn.ReflectionPad2d(40))
            name = "conv-%d" % (idx)

            filter, stride, padding = 3, 1, 0
            if in_dim == 3 or out_dim == 3:
                filter, stride, padding = 9, 1, 4
            elif in_dim == 64 or out_dim == 64:
                filter, stride, padding = 3, 2, 1

            conv = nn.Conv2d(in_dim, out_dim, filter, stride, padding)
            if (in_dim == 28 and out_dim == 64) or (in_dim == 64 and out_dim == 32):
                conv = nn.ConvTranspose2d(in_dim, out_dim, filter, stride, padding, output_padding=1)

            seq.add_module(name, conv)
            self.transform_network.append(seq)
            transform_parameters.append(seq.parameters())

            in_dim = out_dim

        # Download VGG and index layers we're interested in
        self.loss_network = models.vgg19(pretrained=True)
        self.loss_layers = []
        idx, layer_i = 0, 1
        for layer in list(self.loss_network.features):
            losses = ""
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(layer_i)

                if name in self.content_layers:
                    losses += "c"
                if name in self.style_layers:
                    losses += "s"

                if losses != "":
                    self.loss_layers.append((idx, losses))

            if isinstance(layer, nn.ReLU):
                layer_i += 1

            idx += 1

        # Optimization
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(transform_parameters, lr=1e-3)

        if self.use_cuda:
            self.transform_network = [network.cuda() for network in self.transform_network]
            self.loss.cuda()
            self.gram.cuda()

    def train(self, input):
        self.optimizer.zero_grad()

        content = input.clone()
        style = self.style.clone().expand_as(input)

        pastiche = input
        for i in range(len(self.transform_network)):
            layers, out_dim = self.transform_network[i], self.out_dims[i]
            layers.add_module("in_" + str(i), nn.InstanceNorm2d(out_dim))
            layers.add_module("relu_" + str(i), nn.ReLU())

            pastiche = layers(pastiche)

        pastiche.data.clamp_(0, 255)
        pastiche_saved = pastiche.clone()
        
        content_loss = 0
        style_loss = 0

        start_layer = 0
        not_inplace = lambda item: nn.ReLU(inplace=False) if isinstance(item, nn.ReLU) else item
        for layer, losses in self.loss_layers:
            layers = list(self.loss_network.features.children())[start_layer:layer+1]
            layers = [not_inplace(item) for item in layers]

            features = nn.Sequential(*layers)
            if self.use_cuda:
                features.cuda()

            pastiche, content, style = features(pastiche), features(content), features(style)

            if "c" in losses:
                content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)
            if "s" in losses:
                pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                style_loss += self.loss(pastiche_g * self.style_weight, style_g.detach() * self.style_weight)

            start_layer = layer + 1

        total_loss = content_loss + style_loss
        total_loss.backward()

        self.optimizer.step()

        return content_loss, style_loss, pastiche_saved

    def eval(self, input):
        return self.transform_network.forward(input)

    def save(self):
        torch.save(self.transform_network.state_dict(), "models/model")
