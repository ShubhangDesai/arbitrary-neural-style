import torchvision.models as models
import torchvision.datasets as datasets
from losses import *

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_weight = 1
style_weight = 1000

use_cuda = torch.cuda.is_available()

def build_model(content, style):
    cnn = models.vgg19(pretrained=True).features

    # move it to the GPU if possible:
    if use_cuda:
        cnn = cnn.cuda()

    # just in order to have an iterable access to or list of content/syle losses
    content_losses = []
    style_losses = []

    transform_network = nn.Sequential(nn.ReflectionPad2d(40),
                          nn.Conv2d(3, 32, 9, stride=1, padding=4),
                          nn.BatchNorm2d(32),
                          nn.ReLU(),

                          nn.Conv2d(32, 64, 3, stride=2, padding=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),

                          nn.Conv2d(64, 128, 3, stride=2, padding=1),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(128, 128, 3, stride=1,padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),
                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),
                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),
                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),
                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),
                          nn.Conv2d(128, 128, 3, stride=1, padding=0),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),

                          nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                          nn.BatchNorm2d(32),
                          nn.ReLU(),

                          nn.Conv2d(32, 3, 9, stride=1, padding=4),
                          nn.BatchNorm2d(3),
                          nn.ReLU()
                            )

    loss_network = nn.Sequential()
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        loss_network = loss_network.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            loss_network.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = loss_network.forward(content).clone()
                content_loss = ContentLoss(target, content_weight)
                loss_network.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = loss_network.forward(style).clone()
                target_feature_gram = gram.forward(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                loss_network.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            loss_network.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = loss_network.forward(content).clone()
                content_loss = ContentLoss(target, content_weight)
                loss_network.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = loss_network.forward(style).clone()
                target_feature_gram = gram.forward(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                loss_network.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            loss_network.add_module(name, layer)

    return transform_network, loss_network, style_losses, content_losses