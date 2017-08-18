from __future__ import print_function

from StyleCNN import *
from utils import *

# CUDA Configurations
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Content and style
style = image_loader("styles/starry_night.jpg").type(dtype)
content = image_loader("contents/dancing.jpg").type(dtype)
input = image_loader("contents/dancing.jpg").type(dtype)
input.data = torch.randn(input.data.size()).type(dtype)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
num_epochs = 31

def main():
    style_cnn = StyleCNN(style, content, input)

    for i in range(num_epochs):
        pastiche = style_cnn.train()

        if i % 10 == 0:
            print("Iteration: %d" % (i))

            path = "outputs/%d.png" % (i)
            pastiche.data.clamp_(0, 1)
            save_image(pastiche, path)

main()