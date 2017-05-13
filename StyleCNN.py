import torch.optim as optim
import torchvision.models as models

from modules.GramMatrix import *

class StyleCNN(object):
    def __init__(self, style, content, pastiche):
        super(StyleCNN, self).__init__()

        self.style = style
        self.content = content
        self.pastiche = nn.Parameter(pastiche.data)

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000

        self.use_cuda = torch.cuda.is_available()

        self.loss_network = models.vgg19(pretrained=True)

        self.loss_layers = []
        index = 0
        i = 1
        for layer in list(self.loss_network.features):
            losses = ""
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

                if name in self.content_layers:
                    losses += "c"
                if name in self.style_layers:
                    losses += "s"

                if losses != "":
                    self.loss_layers.append((index, losses))

            if isinstance(layer, nn.ReLU):
                i += 1

            index += 1

        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.LBFGS([self.pastiche])

        if self.use_cuda:
            self.loss_network.cuda()
            self.gram.cuda()

    def train(self):
        def closure():
            self.optimizer.zero_grad()

            pastiche = self.pastiche.clone()
            pastiche.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()

            content_loss = 0
            style_loss = 0

            start_layer = 0
            not_inplace = lambda item: nn.ReLU(inplace=False) if isinstance(item, nn.ReLU) else item
            for layer, losses in self.loss_layers:
                layers = list(self.loss_network.features.children())[start_layer:layer+1]
                layers = [not_inplace(item) for item in layers]

                features = nn.Sequential(*layers)
                if self.use_cuda:
                    features.cude()

                pastiche, content, style = features.forward(pastiche), features.forward(content), features.forward(style)

                if "c" in losses:
                    content_loss += self.loss(pastiche, content.detach())
                if "s" in losses:
                    pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                    style_loss += self.loss(pastiche_g, style_g.detach())

                start_layer = layer + 1

            content_loss *= self.content_weight
            style_loss *= self.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()

            return total_loss

        self.optimizer.step(closure)

        return self.pastiche