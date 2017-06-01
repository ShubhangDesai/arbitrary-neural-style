import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import itertools
import scipy.misc

imsize = 256

loader = transforms.Compose([
    transforms.Scale(imsize),
    transforms.CenterCrop(imsize),
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

def get_content_and_style(loader1, loader2, num_iters):
    iter1 = itertools.cycle(loader1)
    iter2 = itertools.cycle(loader2)

    for _ in range(num_iters):
        yield (next(iter1), next(iter2))

