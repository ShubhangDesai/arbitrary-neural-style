import torch.optim as optim
import torchvision.models as models

from modules.GramMatrix import *
from modules.ScaledTanh import *

class StyleCNN(object):
    def __init__(self, style):
        super(StyleCNN, self).__init__()

        self.style = style
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000

        self.use_cuda = torch.cuda.is_available()

        self.transform_network = nn.Sequential(nn.ReflectionPad2d(40),
                              nn.Conv2d(3, 32, 9, stride=1, padding=4),
                              nn.InstanceNorm2d(32, affine=True),
                              nn.ReLU(),

                              nn.Conv2d(32, 64, 3, stride=2, padding=1),
                              nn.InstanceNorm2d(64, affine=True),
                              nn.ReLU(),

                              nn.Conv2d(64, 128, 3, stride=2, padding=1),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),

                              nn.Conv2d(128, 128, 3, stride=1,padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),

                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),

                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),

                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),

                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, 3, stride=1, padding=0),
                              nn.InstanceNorm2d(128, affine=True),
                              nn.ReLU(),

                              nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                              nn.InstanceNorm2d(64, affine=True),
                              nn.ReLU(),

                              nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                              nn.InstanceNorm2d(32, affine=True),
                              nn.ReLU(),

                              nn.Conv2d(32, 3, 9, stride=1, padding=4),
                              nn.InstanceNorm2d(3, affine=True),
                              nn.ReLU()
                                )

        try:
            self.transform_network.load_state_dict(torch.load("models/model"))
        except IOError:
            pass

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
        self.optimizer = optim.Adam(self.transform_network.parameters(), lr=1e-3)

        if self.use_cuda:
            self.transform_network.cuda()
            self.loss_network.cuda()
            self.gram.cuda()

    def train(self, input):
        self.optimizer.zero_grad()

        content = input.clone()
        style = self.style.clone().expand_as(input)
        pastiche = self.transform_network(input)
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
