import os

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

root_picasso = 'data_picasso'
root_photo = 'data_photo'

num_epochs = 3
image_size = 256
batch_size = 4
random_seed = 42
learning_rate = 2e-4
betas_values = (0.5, 0.999)
cycle_lambda = 10
identity_lambda = 0


class PabloPicassoDataset(Dataset):
    def __init__(self, root_picasso, root_photo, transform=None):
        self.root_picasso = root_picasso
        self.root_photo = root_photo
        self.transform = transform

        self.picasso_images = os.listdir(root_picasso)
        self.photo_images = os.listdir(root_photo)

        self.picasso_img_len = len(self.picasso_images)
        self.photo_img_len = len(self.photo_images)
        self.images_len = max(self.picasso_img_len, self.photo_img_len)

    def __len__(self):
        return self.images_len

    def __getitem__(self, idx):
        picasso_img = os.path.join(self.root_picasso, self.picasso_images[idx % self.picasso_img_len])
        photo_img = os.path.join(self.root_photo, self.photo_images[idx % self.photo_img_len])

        picasso_img = np.array(Image.open(picasso_img).convert("RGB"))
        photo_img = np.array(Image.open(photo_img).convert("RGB"))

        if self.transform:
            augmentation = self.transform(image=photo_img, image0=picasso_img) # аугментация в albumentations
            photo_img = augmentation['image']
            picasso_img = augmentation['image0']

        return picasso_img, photo_img


transforms = A.Compose([A.Resize(height=image_size, width=image_size),
                        A.Normalize(mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5),
                                    max_pixel_value=255.0),
                        ToTensorV2()],
                       additional_targets={"image0": "image"})

picasso_images = PabloPicassoDataset(root_picasso=root_picasso,
                                     root_photo=root_photo,
                                     transform=transforms)

loader = DataLoader(dataset=picasso_images,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      4,
                      stride,
                      padding=1,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class DiscriminatorModel(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,
                      features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True))

        layers = []
        in_ch = features[0]

        for out_ch in features[1:]:
            if out_ch == 512:

                layers.append(ConvBlock(in_ch,
                                        out_ch,
                                        stride=1))
            else:
                layers.append(ConvBlock(in_ch,
                                        out_ch))
            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch,
                                out_channels=1,
                                kernel_size=4,
                                stride=1,
                                padding=1,
                                padding_mode="reflect"
                                ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.layers(self.initial(x)))


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, down=True, use_act=True, **kwargs):
        super().__init__()
        if down:
            conv = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             padding_mode='reflect',
                             **kwargs)
        else:
            conv = nn.ConvTranspose2d(in_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=1,
                                      **kwargs)

        norm = nn.InstanceNorm2d(out_channels)
        if use_act:
            act = nn.ReLU(inplace=True)
        else:
            act = nn.Identity()

        self.conv = nn.Sequential(conv, norm, act)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res = nn.Sequential(
            ConvolutionalBlock(channels,
                               channels,
                               stride=1),

            ConvolutionalBlock(channels,
                               channels,
                               stride=1,
                               use_act=False))

    def forward(self, x):
        return x + self.res(x)


class GeneratorModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_resudials=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=7,
                      stride=1,
                      padding=3),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.down = nn.Sequential(
            ConvolutionalBlock(out_channels,
                               out_channels * 2),
            ConvolutionalBlock(out_channels * 2,
                               out_channels * 4))

        self.res = nn.Sequential(*[ResidualBlock(out_channels * 4) for _ in range(num_resudials)])

        self.up = nn.Sequential(
            ConvolutionalBlock(out_channels * 4,
                               out_channels * 2,
                               down=False,
                               output_padding=1),
            ConvolutionalBlock(out_channels * 2,
                               out_channels,
                               down=False,
                               output_padding=1))
        self.last = nn.Conv2d(out_channels,
                              in_channels,
                              kernel_size=7,
                              stride=1,
                              padding=3,
                              padding_mode="reflect"
                              )

    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        x = self.last(x)

        return torch.tanh(x)


disc_ph = DiscriminatorModel().to(device)
disc_p = DiscriminatorModel().to(device)
gen_phtp = GeneratorModel().to(device)
gen_ptph = GeneratorModel().to(device)

opt_disc = optim.Adam(list(disc_ph.parameters()) + list(disc_p.parameters()),
                      lr=learning_rate, betas=betas_values)
opt_gen = optim.Adam(list(gen_phtp.parameters()) + list(gen_ptph.parameters()),
                     lr=learning_rate, betas=betas_values)

l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

G_loss_list = []
D_loss_list = []

torch.cuda.empty_cache()


def train_picasso(gen_ptph, gen_phtp, disc_ph, disc_p, loader, opt_gen, opt_disc, l1_loss, mse_loss, G_loss_list, D_loss_list):
    # Обучение дискриминатора
    loop = tqdm(loader)
    for idx, (picasso_img, photo_img) in enumerate(loop):
        picasso_img = picasso_img.to(device)
        photo_img = photo_img.to(device)

        # Обучим дискриминатор определять изображения картин Пикассо
        fake_picasso = gen_phtp(photo_img)
        D_m_real = disc_p(picasso_img)
        D_m_fake = disc_p(fake_picasso.detach())

        D_m_fake_loss = mse_loss(D_m_fake, torch.zeros_like(D_m_fake))
        D_m_real_loss = mse_loss(D_m_real, torch.ones_like(D_m_real))
        D_m_loss = D_m_fake_loss + D_m_real_loss

        # Обучим дискриминатор определять изображения различных фотографий
        fake_photo = gen_ptph(picasso_img)
        D_p_real = disc_ph(photo_img)
        D_p_fake = disc_ph(fake_photo.detach())

        D_p_real_loss = mse_loss(D_p_real, torch.ones_like(D_p_real))
        D_p_fake_loss = mse_loss(D_p_fake, torch.zeros_like(D_p_fake))
        D_p_loss = D_p_real_loss + D_p_fake_loss

        # Сложим функции потерь двух дискриминаторов
        D_loss = (D_p_loss + D_m_loss) / 2
        D_loss_list.append(D_loss.item())

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Обучим генераторы
        # Adversarial loss
        D_m_fake = disc_p(fake_picasso)
        D_p_fake = disc_ph(fake_photo)

        G_m_loss = mse_loss(D_m_fake, torch.ones_like(D_m_fake))
        G_p_loss = mse_loss(D_p_fake, torch.ones_like(D_p_fake))

        # Cycle_loss
        cycle_picasso = gen_phtp(fake_photo)
        cycle_m_loss = l1_loss(picasso_img, cycle_picasso)

        cycle_photo = gen_ptph(fake_picasso)
        cycle_p_loss = l1_loss(photo_img, cycle_photo)

        cycle_loss = cycle_p_loss + cycle_m_loss

        # Identity loss
        identity_m = gen_phtp(picasso_img)
        identity_m_loss = l1_loss(picasso_img, identity_m)

        identity_p = gen_ptph(photo_img)
        identity_p_loss = l1_loss(photo_img, identity_p)

        identity_loss = identity_m_loss + identity_p_loss

        # Сложим все функции потерь
        G_loss = (G_m_loss +
                  G_p_loss +
                  cycle_lambda * cycle_loss +
                  identity_lambda * identity_loss)
        G_loss_list.append(G_loss.item())

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        # if idx % 200 == 0:
        #    show_img(photo_img, title='Photo')
        #     show_img(gen_ptm(photo_img), title='Photo-to-Monet Translation')

        loop.set_postfix(gen_loss=round(G_loss.item(), 2), dics_loss=round(D_loss.item(), 2))
