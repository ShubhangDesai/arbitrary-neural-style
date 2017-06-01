from __future__ import print_function

import torch.utils.data
import torchvision.datasets as datasets

from StyleCNN import *
from utils import *

# CUDA Configurations
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
num_iters = 40000
N = 4

# Test-time images
content = image_loader("contents/test.jpg").type(dtype)
style = image_loader("styles/udnie.jpg").type(dtype)

def main():
    style_cnn = StyleCNN()
    
    if content is not None and style is not None:
        pastiche = style_cnn.eval(content, style)
        save_images(pastiche, ["outputs/eval.png"])
        return

    # Contents
    coco = datasets.ImageFolder(root='data/contents', transform=loader)
    content_loader = torch.utils.data.DataLoader(coco, batch_size=N, shuffle=True, **kwargs)

    # Styles
    paintings = datasets.ImageFolder(root='data/overfit_styles', transform=loader)
    style_loader = torch.utils.data.DataLoader(paintings, batch_size=1, shuffle=False, **kwargs)

    images = get_content_and_style(content_loader, style_loader, num_iters=num_iters)
    for i, x in enumerate(images):
        i += 34410
        content_batch, style_batch = x[0][0], x[1][0]
        content_batch, style_batch = Variable(content_batch).type(dtype), Variable(style_batch).type(dtype)

        content_loss, style_loss, pastiche = style_cnn.train(content_batch, style_batch)

        if i % 10 == 0:
            print("Iteration: %d" % (i))
            print("Content loss: %f" % (content_loss.data[0]))
            print("Style loss: %f" % (style_loss.data[0]))

        if i % 500 == 0:
            path = "outputs/%d_" % (i)
            paths = [path + str(n) + ".png" for n in range(N)]
            save_images(pastiche, paths)

            path = "outputs/style_%d.png" % (i)
            save_images(style_batch, [path])

            path = "outputs/content_%d_" % (i)
            paths = [path + str(n) + ".png" for n in range(N)]
            save_images(content_batch, paths)
            style_cnn.save()

        if i == num_iters:
            break

main()
