# Note - training loop and architectures are modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py

import argparse
import csv
import random
from pathlib import Path

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.nn.utils.parametrizations import spectral_norm


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)


def get_attributes():
    """
    Read punk attributes file and form one-hot matrix
    """
    df = pd.concat(
        [
            pd.read_csv(f, sep=", ", engine="python")
            for f in Path("attributes").glob("*.csv")
        ]
    )
    accessories = df["accessories"].str.get_dummies(sep=" / ")
    type_ = df["type"].str.get_dummies()
    gender = df["gender"].str.get_dummies()

    return pd.concat([df["id"], accessories, type_, gender], axis=1).set_index("id")


# folder dataset
class Punks(torch.utils.data.Dataset):
    def __init__(self, path, size=10_000):
        self.path = Path(path)
        self.size = size
        self.attributes = get_attributes()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # randomly select attribute
        attribute = random.choice(self.attributes.columns)
        # randomly select punk with that attribute
        id_ = random.choice(self.attributes.index[self.attributes[attribute] == 1])

        return self.transform(
            Image.open(self.path / f"punk{int(id_):03}.png").convert("RGBA")
        )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nc=4, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.network(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)


def main(args):
    Path(args.outf).mkdir(exist_ok=True)

    # for reproducibility
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    cudnn.benchmark = True

    dataset = Punks(args.dataroot)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net_g = Generator(args.nc, args.nz, args.ngf).to(device)
    net_g.apply(weights_init)

    net_d = Discriminator(args.nc, args.ndf).to(device)
    net_d.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    with open(f"{args.outf}/logs.csv", "w") as f:
        csv.writer(f).writerow(["epoch", "loss_g", "loss_d", "d_x", "d_g_z1", "d_g_z2"])

    for epoch in range(args.niter):

        print(f"{epoch}/{args.niter}")
        avg_loss_g = AverageMeter()
        avg_loss_d = AverageMeter()
        avg_d_x = AverageMeter()
        avg_d_g_z1 = AverageMeter()
        avg_d_g_z2 = AverageMeter()

        for data in dataloader:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            net_d.zero_grad()
            real_cpu = data.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full(
                (batch_size,), real_label, dtype=real_cpu.dtype, device=device
            )

            output = net_d(real_cpu)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()

            avg_d_x.update(output.mean().item(), batch_size)

            # train with fake
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = net_g(noise)
            label.fill_(fake_label)
            output = net_d(fake.detach())
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()
            optimizer_d.step()

            avg_loss_d.update((loss_d_real + loss_d_fake).item(), batch_size)
            avg_d_g_z1.update(output.mean().item())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            net_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = net_d(fake)
            # minimize loss but also maximize alpha channel
            loss_g = criterion(output, label) + fake[:, -1].mean()
            loss_g.backward()
            optimizer_g.step()

            avg_loss_g.update(loss_g.item(), batch_size)
            avg_d_g_z2.update(output.mean().item())

        # write logs
        with open(f"{args.outf}/logs.csv", "a") as f:
            csv.writer(f).writerow(
                [epoch, avg_loss_g, avg_loss_d, avg_d_x, avg_d_g_z1, avg_d_g_z2]
            )

        if (epoch + 1) % args.save_every == 0:
            # save samples
            fake = net_g(fixed_noise)
            vutils.save_image(
                fake.detach(),
                f"{args.outf}/fake_samples_epoch_{epoch}.png",
                normalize=True,
            )

            # save_checkpoints
            torch.save(net_g.state_dict(), f"{args.outf}/net_g_epoch_{epoch}.pth")
            torch.save(net_d.state_dict(), f"{args.outf}/net_d_epoch_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data", help="path to dataset")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--nz", type=int, default=100, help="size of the latent z vector"
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument(
        "--niter", type=int, default=1000, help="number of epochs to train for"
    )
    parser.add_argument("--save_every", type=int, default=10, help="how often to save")
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--outf", default="out", help="folder to output images and model checkpoints"
    )
    parser.add_argument("--manual_seed", type=int, default=0, help="manual seed")

    args = parser.parse_args()
    args.cuda = True
    args.nc = 4
    print(args)
    main(args)
