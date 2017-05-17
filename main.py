from __future__ import print_function

import torch.utils.data
import torchvision.datasets as datasets

from StyleCNN import *
from utils import *

# CUDA Configurations
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Content and style
style = image_loader("styles/picasso.jpg").type(dtype)
content = None

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
num_epochs = 2

def main():
    style_cnn = StyleCNN(style)
    coco = datasets.CocoCaptions(root='data/', annFile=None, transform=loader)
    train_loader = torch.utils.data.DataLoader(coco, batch_size=1, shuffle=True, **kwargs)

    if content is not None:
        style_cnn.eval(content)
        return

    iter = 0
    for i in range(num_epochs):
        for x_batch, _ in train_loader:
            input = Variable(x_batch).type(dtype)
            content_loss, style_loss, pastiche = style_cnn.train(input)

            if iter % 10 == 0:
                print("Iteration: %d" % (iter))
                print("Content loss: %f" % (content_loss.data[0]))
                print("Style loss: %f" % (style_loss.data[0]))

                path = "outputs/%d.png" % (iter)
                save_image(pastiche, path)

                style_cnn.save()

            iter += 1

    style_cnn.save()

main()