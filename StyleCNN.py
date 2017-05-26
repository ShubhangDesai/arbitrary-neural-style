import torch.optim as optim
import torchvision.models as models
from torch.nn import Parameter

from modules.Flatten import *
from modules.GramMatrix import *
from modules.ScaledTanh import *
from modules.LearnedInstanceNorm2d import *

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

        self.normalization_network = nn.Sequential(nn.Conv2d(3, 32, 9, stride=2, padding=0),
                                                   nn.Conv2d(32, 64, 9, stride=2, padding=0),
                                                   nn.Conv2d(64, 128, 9, stride=2, padding=0),
                                                   Flatten(),
                                                   nn.Linear(625, 256),
                                                   nn.Linear(256, 32)
                                                   )

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

        self.out_dims = [32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]

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

        self.loss = nn.MSELoss()
        norm_params = torch.FloatTensor(128, 32)
        torch.randn(128, 16, out=norm_params[:, :16]).mul_(0.01).add_(1)
        torch.randn(128, 16, out=norm_params[:, 16:]).mul_(0.01)
        #norm_params[:, 16:].zero_()
        self.norm_params = Parameter(norm_params)
        #self.normalization_optimizer = optim.Adam(self.normalization_network.parameters(), lr=1e-3)
        self.normalization_optimizer = optim.Adam([self.norm_params], lr=1e-3)
        self.transform_optimizer = optim.Adam(self.transform_network.parameters(), lr=1e-3)

        print("test")

        if self.use_cuda:
            self.normalization_network.cuda()
            self.loss.cuda()
            self.gram.cuda()

    def train(self, input):
        self.normalization_network.zero_grad()
        self.transform_optimizer.zero_grad()

        content = input.clone()
        style = self.style.clone()
        pastiche = input
        #norm_params = self.normalization_network.forward(style)
        norm_params = self.norm_params
        N = norm_params.size(1)

        idx = 0
        for layer in list(self.transform_network):
            if idx != 0:
                out_dim = self.out_dims[idx - 1]
                weight = norm_params[:out_dim, idx - 1].data
                bias = norm_params[:out_dim, idx + int(N/2) - 1].data
                #weight = torch.ones(out_dim)
                #bias = torch.zeros(out_dim)
                instance_norm = LearnedInstanceNorm2d(out_dim, Parameter(weight), Parameter(bias))

                layers = nn.Sequential(*[layer, instance_norm, nn.ReLU()])
            else:
                layers = nn.Sequential(layer)
            
            if self.use_cuda:
                layers.cuda()

            pastiche = layers(pastiche)
            idx += 1

        pastiche.data.clamp_(0, 255)
        pastiche_saved = pastiche.clone()
        
        content_loss = 0
        style_loss = 0

        start_layer = 0
        style = style.expand_as(input)
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

        self.normalization_optimizer.step()
        self.transform_optimizer.step()

        return content_loss, style_loss, pastiche_saved

    def eval(self, input):
        return self.transform_network.forward(input)

    def save(self):
        torch.save(self.transform_network.state_dict(), "models/model")

    def norm_test(self):
        return self.norm_params
