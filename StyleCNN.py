import torch.optim as optim
import torchvision.models as models

from modules.GramMatrix import *

class StyleCNN(object):
    def __init__(self, style):
        super(StyleCNN, self).__init__()

        self.style = style

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000

        self.use_cuda = torch.cuda.is_available()

        self.loss_network = models.vgg19(pretrained=True)

        self.transform_network = nn.Sequential(nn.ReflectionPad2d(40),
                                               nn.Conv2d(3, 32, 9, stride=1, padding=4),
                                               nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                               nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                               nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                               nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                                               nn.Conv2d(32, 3, 9, stride=1, padding=4),
                                               )

        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.transform_network.parameters(), lr=1e-3)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.loss_network.cuda()
            self.gram.cuda()

def train(self, content):
        self.optimizer.zero_grad()

        content = content.clone()
        style = self.style.clone()
        pastiche = self.transformation_network.forward(content)

        content_loss = 0
        style_loss = 0

        i = 1
        not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
        for layer in list(self.loss_network.features):
            layer = not_inplace(layer)
            if self.use_cuda:
                layer.cuda()

            pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

                if name in self.content_layers:
                    content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)
                if name in self.style_layers:
                    pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                    style_loss += self.loss(pastiche_g * self.style_weight, style_g.detach() * self.style_weight)

            if isinstance(layer, nn.ReLU):
                i += 1

        total_loss = content_loss + style_loss
        total_loss.backward()

        self.optimizer.step()

        return self.pastiche