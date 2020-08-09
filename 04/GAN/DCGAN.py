import math

import torch
import torchvision
import tqdm

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, models
from torchvision.datasets import FashionMNIST, ImageFolder
from torchvision.utils import save_image

from IPython.display import Image, display_jpeg

USE_CUDA = torch.cuda.is_available()
DISPLAY = torch.device("cuda" if USE_CUDA else "cpu")
print("Current Displaying Device : ", DISPLAY)


class DownSizedPairImageFolder(ImageFolder):
    def __init__(self, root, transform=None,
                 large_size=128, small_size=32, **kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Resize(large_size)
        self.small_resizer = transforms.Resize(small_size)


img_data = ImageFolder(
    "../04/oxford-102/",
    transform=transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])
)

batch_size = 64
img_loader = DataLoader(img_data, batch_size=batch_size,
                        shuffle=True)

nz = 100
ngf = 32

in_size = 1
stride = 1
padding = 0
kernel_size = 4
output_padding = 0

"""
    The meaning of 3 is 3-Color.
    (Maybe RGB(?))

    Create 'z', a Hidden-Vector Variable as 100-Dimension.
    And then construct create-model that generates Image as 3 X 64 X 64
    from 'z', a Hidden-Vector Variable.

    Generate Image-Creation-Model.
"""


class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Convert Image-Size here by Transposed-Convolution.
            nn.ConvTranspose2d(
                nz, ngf * 8,
                4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                ngf * 8, ngf * 4,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                ngf * 4, ngf * 2,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                ngf * 2, ngf,
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                ngf, 3,
                4, 2, 1, bias=False
            ),
            # You can iterate ConvTranspose2d code
            # At this point.
            # out_size = (in_size - 1) * stride - 2 * padding \
            #    + kernel_size + output_padding
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out


"""
    Recognition Model
"""
ndf = 32


class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        out = self.main(x)
        return out.squeeze()


d = DNet().to(DISPLAY)
print(d)

g = GNet().to(DISPLAY)
print(g)

opt_d = optim.Adam(d.parameters(),
                   lr=0.0002, betas=(0.5, 0.999))
opt_g = optim.Adam(g.parameters(),
                   lr=0.0002, betas=(0.5, 0.999))

# Create auxiliary variable that calculate Cross-Entropy.
ones = torch.ones(batch_size).to(DISPLAY)
zeros = torch.zeros(batch_size).to(DISPLAY)
loss_f = nn.BCEWithLogitsLoss()


# For Monitoring, the Variable 'z'.
fixed_z = torch.randn(batch_size, nz, 1, 1).to(DISPLAY)


