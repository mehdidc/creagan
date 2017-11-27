# Like conditional gan paper : https://arxiv.org/abs/1411.1784
# discr takes image and onehot of the class, and produces whether real/fake
# generator takes noise and onehot of the lass, and prodcuces an image.
# generator trained to fool the discr.
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from torch.optim.lr_scheduler import StepLR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', required=True, help='number of classes')
    parser.add_argument('--nc', required=True, default=3, help='number of color channels')
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw |mnist ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

    opt = parser.parse_args()
    print(opt)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(
            root=opt.dataroot,
            transform=transforms.Compose([
               transforms.Scale(opt.imageSize),
               transforms.CenterCrop(opt.imageSize),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(
            root=opt.dataroot,
            download=True,
            transform=transforms.Compose([
                transforms.Scale(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(
            db_path=opt.dataroot, classes=['bedroom_train'],
            transform=transforms.Compose([
                transforms.Scale(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        )
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(
            root=opt.dataroot, download=True,
            transform=transforms.Compose([
                transforms.Scale(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    else:
        raise ValueError(opt.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchSize,
        shuffle=True, 
        num_workers=int(opt.workers)
    )

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    nb_classes = int(opt.classes)
    nb_initial_classes = 10
    nb_ext_classes = nb_classes - nb_initial_classes

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    class _netG(nn.Module):
        def __init__(self, ngpu):
            super(_netG, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz + nb_classes, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        def forward(self, input):
            gpu_ids = None
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)

    netG = _netG(ngpu)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    class _netD(nn.Module):
        def __init__(self, ngpu):
            super(_netD, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8,  1 + nb_classes, 4, 1, 0, bias=False),
            )
        def forward(self, input):
            gpu_ids = None
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
            return output.view(-1, 1 + nb_classes)

    netD = _netD(ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

    nb_rows = 10
    fixed_z = torch.randn(nb_rows, nb_classes,        nz, 1, 1)
    fixed_z = fixed_z.view(nb_rows * nb_classes, nz, 1, 1)
    fixed_onehot = torch.zeros(nb_rows, nb_classes, nb_classes, 1, 1)
    fixed_onehot = fixed_onehot.view(nb_rows * nb_classes, nb_classes, 1, 1)
    for i in range(fixed_onehot.size(0)):
        cl = i % nb_classes
        fixed_onehot[i, cl] = 1
    fixed_noise = torch.cat((fixed_z, fixed_onehot), 1).cuda()
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    
    aux_criterion = nn.CrossEntropyLoss().cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

    schedulerD = StepLR(optimizerD, step_size=100, gamma=0.1)
    schedulerG = StepLR(optimizerG, step_size=100, gamma=0.1)
     
    phi = 0.5
    for epoch in range(opt.niter):
        schedulerD.step()
        schedulerG.step()
        for i, data in enumerate(dataloader):
            # A) optimize discriminator
            netD.zero_grad()
            X, Y = data
            Y = Y.long()
            Y_vect = Y
            Y = Variable(Y)
            Y = Y.cuda()

            batch_size = X.size(0)
            
            # 1) backprop D through real images
            input.data.resize_(X.size()).copy_(X)
            label.data.resize_(batch_size).fill_(real_label)
    
            output = netD(input)
            D = output[:, 0]
            C = output[:, 1:]
            errD_real = (
                criterion(nn.Sigmoid()(D), label) + 
                phi * aux_criterion(C, Y)
            )
            _, pred = C.max(1)
            acc_real = torch.mean((pred.data.cpu() == Y_vect).float())

            errD_real.backward()
            D_x = output.data.mean()

            # 2) backprop D through fake images

            nb_per_class = batch_size // nb_initial_classes
            nb_ext = nb_per_class * nb_ext_classes
            Y_ext = torch.rand(nb_ext, nb_ext_classes) 
            _, Y_ext = Y_ext.max(1)
            Y_ext = Y_ext + nb_initial_classes
            batch_size += nb_ext
            Y_vect = torch.cat((Y_vect, Y_ext), 0)
            Y = Variable(Y_vect).cuda()

            Y_onehot = torch.zeros(batch_size, nb_classes)
            Y_onehot.scatter_(1, Y_vect.view(-1, 1), 1)
            z = torch.randn(batch_size, nz, 1, 1)
            z = torch.cat((z, Y_onehot), 1)
            noise.data.resize_(z.size()).copy_(z)
            fake = netG(noise)
            
            label.data.resize_(batch_size).fill_(fake_label)
            output = netD(fake.detach())
            D = output[:, 0]
            C = output[:, 1:]
            errD_fake = (
                criterion(nn.Sigmoid()(D), label) + 
                phi * aux_criterion(C, Y)
            )

            _, pred = output[:, 1:].max(1)
            acc_fake = torch.mean((pred.data.cpu() == Y_vect).float())

            errD_fake.backward()

            # 3) update params of discriminator
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # B) Optimize generator
            netG.zero_grad()
            label.data.fill_(real_label)
            output = netD(fake)
            D = output[:, 0]
            C = output[:, 1:]
            errG = (
                criterion(nn.Sigmoid()(D), label) + 
                phi * aux_criterion(C, Y)
            )
            errG.backward()
            optimizerG.step()
            fmt = '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f acc_real : %.4f acc_fake : %.4f '
            print(fmt % (epoch, opt.niter, i, len(dataloader), errD.data[0], errG.data[0], acc_real, acc_fake))
            if i % 100 == 0:
                # the first 64 samples from the mini-batch are saved.
                vutils.save_image((X[0:64,:,:,:]+1)/2., '%s/real_samples.png' % opt.outf)
                fake = netG(fixed_noise)
                im = (fake.data + 1) / 2.
                fname = '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
                vutils.save_image(im, fname, nrow=nb_classes)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
