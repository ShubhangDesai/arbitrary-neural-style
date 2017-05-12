from __future__ import print_function

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from model import build_model
from utils import image_loader
from utils import save_image

# CUDA Configurations
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Content and style styles
style = image_loader("styles/picasso.jpg").type(dtype)
content = image_loader("contents/dancing.jpg").type(dtype)

losses = {"style": [], "content": []}

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def train(model, loss_network, batch_loader, num_epochs=2):
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    iteration = [0]
    for i in range(num_epochs):
        for x_batch, _ in batch_loader:
            if iteration[0] == 300:
                break

            def closure():
                # correct the values of updated input image
                #x_batch.data.clamp_(0, 1)
                x_var = Variable(x_batch)

                optimizer.zero_grad()
                pastiche = model.forward(x_var)

                loss_network.forward(pastiche)
                style_score = 0
                content_score = 0

                for sl in losses["style"]:
                    style_score += sl.backward()
                for cl in losses["content"]:
                    content_score += cl.backward()

                iteration[0] += 1
                if iteration[0] % 10 == 0:
                    print("iteration " + str(iteration) + ":")
                    print(style_score.data[0])
                    print(content_score.data[0])

                    #path = 'outputs/%d.png' % (iteration[0])
                    #save_image(pastiche, path)

                return content_score + style_score

            optimizer.step(closure)

    #input.data.clamp_(0, 1)
    #return input

def eval(model, loss_network, batch_loader):
    for x_batch, _ in batch_loader:
        x_var = Variable(x_batch)

        pastiche = model.forward(x_var)

        loss_network.forward(pastiche)
        style_score = 0
        content_score = 0

        for sl in losses["style"]:
            style_score += sl.backward()
        for cl in losses["content"]:
            content_score += cl.backward()

        print(style_score.data[0])
        print(content_score.data[0])

        save_image(x_var, "outputs/test1.png")
        save_image(pastiche, "outputs/test2.png")

        break


def main():
    # Load and build CNN
    _, loss_network, losses["style"], losses["content"] = build_model(content, style)
    model = torch.load("models/model")

    cifar100 = datasets.CIFAR100('data', download=True,
                                 transform=transforms.Compose([
                                     transforms.Scale(256),
                                     transforms.ToTensor()
                                 ]))
    train_loader = torch.utils.data.DataLoader(cifar100, batch_size=1, shuffle=True, **kwargs)

    # Input image
    # input = image_loader("contents/dancing.jpg").type(dtype)
    # input.data = torch.randn(input.data.size()).type(dtype)

    #train(model, loss_network, train_loader)
    eval(model, loss_network, train_loader)
    torch.save(model, "models/model")

    # save final image
    #save_image(pastiche, 'outputs/final.png')

main()