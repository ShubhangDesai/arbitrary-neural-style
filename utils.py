import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image

import scipy.misc

imsize = 256

loader = transforms.Compose([
    transforms.Scale(imsize),
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

def save_images(input, paths):
    N = input.size()[0]
    images = input.data.clone().cpu()
    for n in range(N):
        image = images[n]
        image = image.view(3, imsize, imsize)
        image = unloader(image)
        scipy.misc.imsave(paths[n], image)