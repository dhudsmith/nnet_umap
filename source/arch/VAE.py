import torch
import torch.nn as nn
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, coarse_resolution=(4, 4), hid_channels=(8, 12, 16)):
        super(Encoder, self).__init__()

        self.coarse_resolution = coarse_resolution

        # block 1:
        self.block1 = nn.Sequential(OrderedDict(
            conv=nn.Conv2d(in_channels=1, out_channels=hid_channels[0], kernel_size=5, stride=1, padding=2, bias=False),
            bn=nn.BatchNorm2d(hid_channels[0]),
            relu=nn.ReLU(),
            pool=nn.MaxPool2d(2)
        ))

        # block 2:
        self.block2 = nn.Sequential(OrderedDict(
            conv=nn.Conv2d(in_channels=hid_channels[0], out_channels=hid_channels[1], kernel_size=3, stride=1,
                           padding=1, bias=False),
            bn=nn.BatchNorm2d(num_features=hid_channels[1]),
            relu=nn.ReLU(),
            pool=nn.MaxPool2d(2)
        ))

        # block 3:
        self.block3 = nn.Sequential(OrderedDict(
            conv=nn.Conv2d(in_channels=hid_channels[1], out_channels=hid_channels[2], kernel_size=3, stride=1,
                           padding=1, bias=False),
            bn=nn.BatchNorm2d(num_features=hid_channels[2]),
            relu=nn.ReLU(),
            pool=nn.AdaptiveAvgPool2d(output_size=self.coarse_resolution),
            flatten=nn.Flatten()
        ))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        z = self.block3(x)

        return (z)


class Decoder(nn.Module):
    def __init__(self, coarse_resolution=(4, 4), hid_channels=(16, 12, 8)):
        super(Decoder, self).__init__()

        self.coarse_resolution = coarse_resolution
        self.zdim = hid_channels[0] * self.coarse_resolution[0] * self.coarse_resolution[1]

        # block 1:
        self.block1 = nn.Sequential(OrderedDict(
            unflatten=nn.Unflatten(-1, (hid_channels[0], self.coarse_resolution[0], self.coarse_resolution[1])),
            conv=nn.Conv2d(in_channels=hid_channels[0], out_channels=hid_channels[1], kernel_size=3, stride=1,
                           padding=1, bias=False),
            bn=nn.BatchNorm2d(hid_channels[1]),
            relu=nn.ReLU(),
            upsample=nn.UpsamplingBilinear2d(scale_factor=2)
        ))

        # block 2:
        self.block2 = nn.Sequential(OrderedDict(
            conv=nn.Conv2d(in_channels=hid_channels[1], out_channels=hid_channels[2], kernel_size=3, stride=1,
                           padding=1, bias=False),
            bn=nn.BatchNorm2d(num_features=hid_channels[2]),
            relu=nn.ReLU(),
            upsample=nn.UpsamplingBilinear2d(scale_factor=2)
        ))

        # block 3:
        self.block3 = nn.Sequential(OrderedDict(
            conv=nn.Conv2d(in_channels=hid_channels[2], out_channels=hid_channels[2], kernel_size=3, stride=1,
                           padding=1, bias=False),
            bn=nn.BatchNorm2d(num_features=hid_channels[2]),
            relu=nn.ReLU(),
            upsample=nn.UpsamplingBilinear2d(size=(28, 28))
        ))

        # touch-up
        self.touchup = nn.Sequential(OrderedDict(
            conv=nn.Conv2d(in_channels=hid_channels[2], out_channels=1, kernel_size=3, stride=1, padding=1),
            sig=nn.Sigmoid()
        ))

    def forward(self, z):
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        x = self.touchup(z)

        return (x)


class VAE(nn.Module):
    def __init__(self,
                 coarse_resolution=(4, 4),
                 hid_channels_encoder=(8, 12, 16),
                 hid_channels_decoder=(16, 12, 8),
                 hdim=128):
        super(VAE, self).__init__()

        self.coarse_resolution = coarse_resolution
        self.hid_channels_encoder = hid_channels_encoder
        self.hid_channels_decoder = hid_channels_decoder
        self.hdim = hdim
        self.zdim = coarse_resolution[0] * coarse_resolution[1] * hid_channels_encoder[2]

        # encoder/decoder
        self.encoder = Encoder(coarse_resolution, hid_channels_encoder)
        self.decoder = Decoder(coarse_resolution, hid_channels_decoder)

        # reparametrization
        self.mu = nn.Linear(self.zdim, self.hdim)
        self.logvar = nn.Linear(self.zdim, self.hdim)
        self.zdecode = nn.Linear(self.hdim, self.zdim)

    def encode(self, x):
        z = self.encoder(x)
        return self.mu(z), self.logvar(z)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, h):
        z = self.zdecode(h)
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        h = self.sample(mu, log_var)
        return self.decode(h), mu, log_var

