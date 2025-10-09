import os
import numpy as np
import random
import re
import argparse
import time
import json
import scipy.misc
import itertools
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import init
from torchvision import transforms, models
from PIL import Image
import math
from visdom import Visdom
from tqdm import tqdm
import matplotlib.pyplot as plt


# ==================== DATALOADER ====================
class GANDataset(Dataset):
    def __init__(self, rootA, rootB, transform=None, unaligned=False, device=None, test=False):
        # Natural sorting that handles numbers correctly
        def natural_sort_key(name):
            return [int(text) if text.isdigit() else text.lower() 
                   for text in re.split('([0-9]+)', name)]
        
        sortedA = sorted(os.listdir(rootA), key=natural_sort_key)
        sortedB = sorted(os.listdir(rootB), key=natural_sort_key)
        
        self.image_pathsA = list(map(lambda x: os.path.join(rootA, x), sortedA))
        self.image_pathsB = list(map(lambda x: os.path.join(rootB, x), sortedB))

        self.transform = transform
        self.unaligned = unaligned
        self.device = device
        self.test = test

    def __getitem__(self, index):
        image_pathA = self.image_pathsA[index]
        imageA = Image.open(image_pathA).convert('RGB')

        if self.unaligned:
            image_pathB = self.image_pathsB[random.randint(0, len(self.image_pathsB)-1)]
        else:
            image_pathB = self.image_pathsB[index]

        imageB = Image.open(image_pathB).convert('RGB')

        if self.transform is not None:
            if self.test:
                imageA = self.transform(imageA)
                imageB = self.transform(imageB)
            else:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                imageA = self.transform(imageA)
                random.seed(seed)
                imageB = self.transform(imageB)

        if self.device is not None:
            imageA = imageA.to(self.device)
            imageB = imageB.to(self.device)

        return imageA, imageB, index+1

    def __len__(self):
        return max(len(self.image_pathsA), len(self.image_pathsB))
def get_dataloader(image_pathA, image_pathB, batch_size, resize, crop, unaligned=False, device=None, shuffle=True, test=False):
    if test:
        transform = transforms.Compose([
            transforms.Resize(crop, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize, Image.BICUBIC),
            transforms.RandomCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

    batch_dataset = GANDataset(image_pathA, image_pathB, transform, unaligned, device, test)
    return DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=shuffle)

# ==================== MODELS ====================
# unet
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=False,
                 do_norm=True, norm='batch', do_activation=True):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                             padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.leakyReLU = nn.LeakyReLU(0.2, True)
        self.do_norm = do_norm  # Store the parameter
        self.do_activation = do_activation
        
        if do_norm:
            if norm == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm == 'none':
                self.do_norm = False
            else:
                raise NotImplementedError("norm error")
        else:
            self.norm = None  # Ensure norm is defined

    def forward(self, x):
        if self.do_activation:
            x = self.leakyReLU(x)
        x = self.conv(x)
        if self.do_norm and self.norm is not None:  # Modified condition
            x = self.norm(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False,
                 do_norm=True, norm='batch', do_activation=True, dropout_prob=0.0):
        super(DecoderBlock, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                       stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.drop = nn.Dropout2d(dropout_prob)
        self.do_activation = do_activation
        self.do_norm = do_norm  # ADD THIS LINE - store the do_norm parameter
        
        if do_norm:
            if norm == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm == 'none':
                self.do_norm = False
            else:
                raise NotImplementedError("norm error")
        else:
            self.norm = None  # ADD THIS LINE - ensure norm is defined

    def forward(self, x):
        if self.do_activation:
            x = self.relu(x)
        x = self.convT(x)
        if self.do_norm and self.norm is not None:  # MODIFIED THIS LINE
            x = self.norm(x)
        if self.dropout_prob != 0:
            x = self.drop(x)
        return x
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, bias=False, dropout_prob=0.5, norm='batch'):
        super(Generator, self).__init__()

        # 8-step encode
        self.encoder1 = EncoderBlock(in_channels, 64, bias=bias, do_norm=False, do_activation=False)
        self.encoder2 = EncoderBlock(64, 128, bias=bias, norm=norm)
        self.encoder3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.encoder4 = EncoderBlock(256, 512, bias=bias, norm=norm)
        self.encoder5 = EncoderBlock(512, 512, bias=bias, norm=norm)
        self.encoder6 = EncoderBlock(512, 512, bias=bias, norm=norm)
        self.encoder7 = EncoderBlock(512, 512, bias=bias, norm=norm)
        self.encoder8 = EncoderBlock(512, 512, bias=bias, do_norm=False)

        # 8-step decoder
        self.decoder1 = DecoderBlock(512, 512, bias=bias, norm=norm)
        self.decoder2 = DecoderBlock(1024, 512, bias=bias, norm=norm, dropout_prob=dropout_prob)
        self.decoder3 = DecoderBlock(1024, 512, bias=bias, norm=norm, dropout_prob=dropout_prob)
        self.decoder4 = DecoderBlock(1024, 512, bias=bias, norm=norm, dropout_prob=dropout_prob)
        self.decoder5 = DecoderBlock(1024, 256, bias=bias, norm=norm)
        self.decoder6 = DecoderBlock(512, 128, bias=bias, norm=norm)
        self.decoder7 = DecoderBlock(256, 64, bias=bias, norm=norm)
        self.decoder8 = DecoderBlock(128, out_channels, bias=bias, do_norm=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 8-step encoder - FIXED: each encoder should be called separately
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(encode1)
        encode3 = self.encoder3(encode2)
        encode4 = self.encoder4(encode3)
        encode5 = self.encoder5(encode4)
        encode6 = self.encoder6(encode5)
        encode7 = self.encoder7(encode6)
        encode8 = self.encoder8(encode7)

        # 8-step decoder
        decode1 = torch.cat([self.decoder1(encode8), encode7], 1)
        decode2 = torch.cat([self.decoder2(decode1), encode6], 1)
        decode3 = torch.cat([self.decoder3(decode2), encode5], 1)
        decode4 = torch.cat([self.decoder4(decode3), encode4], 1)
        decode5 = torch.cat([self.decoder5(decode4), encode3], 1)
        decode6 = torch.cat([self.decoder6(decode5), encode2], 1)
        decode7 = torch.cat([self.decoder7(decode6), encode1], 1)
        decode8 = self.decoder8(decode7)
        final = self.tanh(decode8)
        return final

# resnet9 (Johnson style)
class ResidualBlock2(nn.Module):
    def __init__(self, in_features, norm_layer=nn.InstanceNorm2d):
        super(ResidualBlock2, self).__init__()
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)
                     ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
class GeneratorJohnson2(nn.Module):
    def __init__(self, image_channel=3, norm='instance', n_res_blocks=9):
        super(GeneratorJohnson2, self).__init__()
        if norm == 'batchnorm':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise Exception("Norm not specified!")

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(image_channel, 64, 7),
                 norm_layer(64),
                 nn.ReLU(inplace=True)
                ]
        in_channels = 64
        out_channels = in_channels * 2
        for i in range(2):
            model += [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                      norm_layer(out_channels),
                      nn.ReLU(inplace=True)
                     ]
            in_channels = out_channels
            out_channels = in_channels * 2

        for i in range(n_res_blocks):
            model += [ResidualBlock2(in_channels, norm_layer=norm_layer)]
            
        out_channels = in_channels // 2
        for i in range(2):
            model += [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                      norm_layer(out_channels),
                      nn.ReLU(inplace=True)
                     ]
            in_channels = out_channels
            out_channels = in_channels // 2

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, 7),
                  nn.Tanh()
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# DenseNet Generator
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4, norm='batch', bias=False):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=bias),
                nn.BatchNorm2d(growth_rate) if norm == 'batch' else nn.InstanceNorm2d(growth_rate),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)
class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm='batch', bias=False):
        super(TransitionDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels) if norm == 'batch' else nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
    
    def forward(self, x):
        return self.conv(x)
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm='batch', bias=False):
        super(TransitionUp, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1, bias=bias),
            nn.BatchNorm2d(out_channels) if norm == 'batch' else nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
class DenseNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, growth_rate=32, compression=0.5, 
                 num_dense_blocks=3, num_layers_per_block=4, norm='batch', bias=False):
        super(DenseNetGenerator, self).__init__()
        
        self.num_dense_blocks = num_dense_blocks
        self.growth_rate = growth_rate
        self.num_layers_per_block = num_layers_per_block
        self.compression = compression
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=bias),
            nn.BatchNorm2d(64) if norm == 'batch' else nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.transition_downs = nn.ModuleList()
        
        in_channels_enc = 64
        encoder_channels = []  # Store output channels at each level
        
        for i in range(num_dense_blocks):
            # Dense block
            dense_block = DenseBlock(in_channels_enc, growth_rate, num_layers_per_block, norm, bias)
            self.encoder_blocks.append(dense_block)
            
            # Calculate output channels after dense block
            out_channels_dense = in_channels_enc + growth_rate * num_layers_per_block
            encoder_channels.append(out_channels_dense)
            
            # Transition down (except for last block)
            if i < num_dense_blocks - 1:
                out_channels_trans = int(out_channels_dense * compression)
                transition = TransitionDown(out_channels_dense, out_channels_trans, norm, bias)
                self.transition_downs.append(transition)
                in_channels_enc = out_channels_trans
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.transition_ups = nn.ModuleList()
        
        # Bottleneck (last encoder block output)
        bottleneck_channels = encoder_channels[-1]
        
        for i in range(num_dense_blocks - 1):
            # Transition up
            in_channels_dec = bottleneck_channels if i == 0 else decoder_channels
            out_channels_trans_up = int(in_channels_dec * compression)
            
            transition_up = TransitionUp(in_channels_dec, out_channels_trans_up, norm, bias)
            self.transition_ups.append(transition_up)
            
            # The input to the decoder dense block is the concatenation of:
            # 1. The upsampled feature map
            # 2. The corresponding encoder feature map (skip connection)
            decoder_in_channels = out_channels_trans_up + encoder_channels[-(i+2)]
            
            # Dense block for decoder
            dense_block_dec = DenseBlock(decoder_in_channels, growth_rate, num_layers_per_block, norm, bias)
            self.decoder_blocks.append(dense_block_dec)
            
            # Update decoder channels for next iteration
            decoder_channels = decoder_in_channels + growth_rate * num_layers_per_block
        
        # Final convolution - adjust input channels based on the last decoder output
        final_in_channels = decoder_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, out_channels, 1, bias=bias),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        encoder_features = []
        
        # Encoder path
        for i in range(self.num_dense_blocks):
            x = self.encoder_blocks[i](x)
            encoder_features.append(x)
            if i < len(self.transition_downs):
                x = self.transition_downs[i](x)
        
        # Decoder path
        decoder_output = encoder_features[-1]  # Start with bottleneck
        
        for i in range(len(self.transition_ups)):
            # Upsample
            decoder_output = self.transition_ups[i](decoder_output)
            
            # Skip connection - get corresponding encoder feature
            skip_feature = encoder_features[-(i+2)]
            
            # Ensure spatial dimensions match (in case of size mismatches)
            if decoder_output.size()[2:] != skip_feature.size()[2:]:
                decoder_output = F.interpolate(decoder_output, size=skip_feature.size()[2:], mode='bilinear', align_corners=False)
            
            # Concatenate along channel dimension
            x_cat = torch.cat([decoder_output, skip_feature], 1)
            
            # Dense block
            decoder_output = self.decoder_blocks[i](x_cat)
        
        # Final convolution
        return self.final_conv(decoder_output)

# patchGAN
class Discriminator(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, bias=False, norm='batch', sigmoid=True):
        super(Discriminator, self).__init__()
        self.sigmoid = sigmoid

        # 70x70 discriminator
        self.disc1 = EncoderBlock(in_channel * 2, 64, bias=bias, do_norm=False, do_activation=False)
        self.disc2 = EncoderBlock(64, 128, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.disc4 = EncoderBlock(256, 512, bias=bias, norm=norm, stride=1)
        self.disc5 = EncoderBlock(512, out_channel, bias=bias, stride=1, do_norm=False)
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, x, ref):
        d1 = self.disc1(torch.cat([x, ref], 1))
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        d5 = self.disc5(d4)
        if self.sigmoid:
            final = self.sigmoid_act(d5)
        else:
            final = d5
        return final

# imageGAN
class Discriminator286(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bias=False, norm='batch', sigmoid=True):
        super(Discriminator286, self).__init__()
        self.sigmoid = sigmoid

        # 286x286 discriminator
        self.disc1 = EncoderBlock(in_channels * 2, 64, bias=bias, do_norm=False, do_activation=False)
        self.disc2 = EncoderBlock(64, 128, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.disc4 = EncoderBlock(256, 512, bias=bias, norm=norm)
        self.disc5 = EncoderBlock(512, 512, bias=bias, norm=norm)
        self.disc6 = EncoderBlock(512, 512, bias=bias, norm=norm, stride=1)
        self.disc7 = EncoderBlock(512, out_channels, bias=bias, stride=1, do_norm=False)
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, x, ref):
        d1 = self.disc1(torch.cat([x, ref], 1))
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        d5 = self.disc5(d4)
        d6 = self.disc6(d5)
        d7 = self.disc7(d6)
        if self.sigmoid:
            final = self.sigmoid_act(d7)
        else:
            final = d7
        return final

# ==================== C-GAN MODEL ====================
class GANModel:
    def __init__(self, args):
        self.start_epoch = 0
        self.args = args

        # In the __init__ method of GANModel class:
        if args.G == 'unet':
            self.G = Generator(bias=args.bias, norm=args.norm, dropout_prob=args.dropout)
        elif args.G == 'densenet':  # NEW: Add DenseNet option
            self.G = DenseNetGenerator(bias=args.bias, norm=args.norm)
        elif args.G == 'resnet9':
            self.G = GeneratorJohnson2()
        else:
            raise NotImplementedError("Wrong G")

        sigmoid = (args.gan_loss == 'BCE')
        if args.D == 'patch':
            self.D = Discriminator(bias=args.bias, norm=args.norm, sigmoid=sigmoid)
        elif args.D == 'image':
            self.D = Discriminator286(bias=args.bias, norm=args.norm, sigmoid=sigmoid)
        else:
            raise NotImplementedError("Wrong D")

        self.init_type = args.init_type
        if args.init_type is not None:
            self.G.apply(self.init_weights)
            self.D.apply(self.init_weights)

        self.optimizer_G = torch.optim.Adam(self.G.parameters(),lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=self.lr_lambda)
        
        if args.gan_loss == 'BCE':
            self.gan_loss_fn = torch.nn.BCELoss()
        elif args.gan_loss == 'MSE':
            self.gan_loss_fn = torch.nnMSELoss()
        else:
            raise NotImplementedError("GAN loss function error")

        self.L1_loss_fn = torch.nn.L1Loss()

        self.lambd = args.lambd
        self.lambd_d = args.lambd_d

        self.d_update_frequency = args.d_update_frequency
    def lr_lambda(self, epoch):
        return 1.0 - max(0, epoch + self.start_epoch - self.args.lr_decay_start) / (self.args.lr_decay_n + 1)    
    def init_weights(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if self.init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif self.init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif self.init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] not implemented' % self.init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.weight.data, 0.0)
    def update_scheduler(self):
        self.scheduler_G.step()
        self.scheduler_D.step()
        print('learning rate = %.7f' % self.optimizer_G.param_groups[0]['lr'])
    def d_update(self, d_loss, epoch):
        if epoch%self.d_update_frequency == 0:
            d_loss.backward()
            self.optimizer_D.step()
    def set_start_epoch(self, epoch):
        self.start_epoch = epoch
    def to(self, device):
        self.G.to(device)
        self.D.to(device)

        for state in itertools.chain(self.optimizer_G.state.values(), self.optimizer_D.state.values()):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    def train(self, input, save, out_dir_img, epoch, i):
        x, y, img_idx = input
        
        # img_idx is a batch of indices, we need to handle them individually if saving
        if save and torch.is_tensor(img_idx):
            # For the first image in the batch, use the first index
            img_idx_val = img_idx[0].item() if len(img_idx) > 0 else 0
        else:
            img_idx_val = 0
        
        #D loss
        self.optimizer_D.zero_grad()
        gen = self.G(x)
        loss_D_real = self.gan_loss(self.D(y, x), 1) * self.lambd_d
        loss_D_fake = self.gan_loss(self.D(gen.detach(), x), 0) * self.lambd_d
        loss_D = loss_D_real + loss_D_fake
        self.d_update(loss_D, i)
        #G loss
        self.optimizer_G.zero_grad()
        loss_G_gan = self.gan_loss(self.D(gen, x), 1) 
        loss_G_L1 = self.L1_loss_fn(gen, y) * self.lambd
        loss_G = loss_G_gan + loss_G_L1
        loss_G.backward()
        self.optimizer_G.step()

        if save:
            self.save_image((x, y, gen), out_dir_img, "train_ep_%d_img_%d" % (epoch, img_idx_val))

        return {'G': loss_G, 'G_gan': loss_G_gan, 'G_L1': loss_G_L1, 
                'D': loss_D, 'D_real': loss_D_real, 'D_fake': loss_D_fake}
    def eval(self, input, save, out_dir_img, epoch):
        with torch.no_grad():
            x, y, img_idx = input
            
            # img_idx is a batch of indices, we need to handle them individually if saving
            if save and torch.is_tensor(img_idx):
                img_idx_val = img_idx[0].item() if len(img_idx) > 0 else 0
            else:
                img_idx_val = 0
                
            gen = self.G(x)

            #D loss
            loss_D_real = self.gan_loss(self.D(y, x), 1) * self.lambd_d
            loss_D_fake = self.gan_loss(self.D(gen, x), 0) / self.lambd_d
            loss_D = loss_D_real + loss_D_fake
            #G loss
            loss_G_gan = self.gan_loss(self.D(gen, x), 1)
            loss_G_L1 = self.L1_loss_fn(gen, y) * self.lambd
            loss_G = loss_G_gan + loss_G_L1
        if save:
            self.save_image((x, y, gen), out_dir_img, "val_ep_%d_img_%d" % (epoch, img_idx_val))
        return {'G': loss_G, 'G_gan': loss_G_gan, 'G_L1': loss_G_L1,
                'D': loss_D, 'D_real': loss_D_real, 'D_fake': loss_D_fake}
    def test(self, images, i, out_dir_img):
        with torch.no_grad():
            A, B, img_idx = images
            
            # img_idx is a batch of indices, we need to handle them individually
            if torch.is_tensor(img_idx):
                img_idx_val = img_idx[0].item() if len(img_idx) > 0 else 0
            else:
                img_idx_val = 0
                
            gen = self.G(A)
            score_gen = self.D(gen, A).mean()
            score_gt = self.D(B, A).mean()
            self.save_image((A, B, gen), out_dir_img, "test_%d" % img_idx_val, test=True)
        return score_gen, score_gt
    def gan_loss(self, out, label):
            return self.gan_loss_fn(out, torch.ones_like(out) if label else torch.zeros_like(out))
    def load_state(self, state, lr=None):
            print('Using pretrained model...')
            self.G.load_state_dict(state['G'])
            self.D.load_state_dict(state['D'])
            self.optimizer_G.load_state_dict(state['optimG'])
            self.optimizer_D.load_state_dict(state['optimD'])
            if lr is not None:
                for param_group in self.optimizer_G.param_groups:
                    before = param_group['lr']
                    param_group['lr'] = lr
                for param_group in self.optimizer_D.param_groups:
                    befors = param_group['lr']
                    param_group['lr'] = lr
                print('optim lr: before={} / after={}'.format(before, lr))
    def save_state(self):
            return {'G': self.G.state_dict(),
                    'D': self.D.state_dict(),
                    'optimG': self.optimizer_G.state_dict(),
                    'optimD': self.optimizer_D.state_dict()}
    def save_image(self, input, filepath, fname, test=False):
            A, B, gen = input
            if test:
                img = self.tensor2image(gen)
                path = os.path.join(filepath, '%s.png' % fname)
                # Replace scipy.misc.imsave with PIL
                img_pil = Image.fromarray(img.squeeze().transpose(1, 2, 0))
                img_pil.save(path)
            else:
                merged = self.tensor2image(self.merge_images(A, B, gen))
                path = os.path.join(filepath, '%s.png' % fname)
                # Replace scipy.misc.imsave with PIL
                merged_pil = Image.fromarray(merged)
                merged_pil.save(path)
            print('saved %s' % path)
    def tensor2image(self, input):
            image_data = input.data
            image = 127.5 * (image_data.cpu().float().numpy() + 1.0)
            return image.astype(np.uint8)
    def merge_images(self, sources, targets, generated):
            row, _, h,w = sources.size()
            merged = torch.zeros([3, row * h, w * 3])
            for idx, (s, t, g) in enumerate(zip(sources, targets, generated)):
                i = idx
                merged[:, i * h:(i+1) * h, 0:w] = s
                merged[:, i * h:(i + 1) * h, w:2 * w] = g
                merged[:, i * h:(i + 1) * h, 2 * w:3 * w] = t
            return merged.permute(1, 2, 0)


def create_plots(stats, out_dir):
    epochs = range(1, len(stats['train_loss']) + 1)
    
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, stats['train_loss'], 'b-', label='Training Loss')
    if stats['val_loss']:
        plt.plot(epochs, stats['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot (you might need to modify this based on your accuracy metric)
    plt.subplot(1, 2, 2)
    if stats.get('train_acc'):
        plt.plot(epochs, stats['train_acc'], 'b-', label='Training Accuracy')
    if stats.get('val_acc'):
        plt.plot(epochs, stats['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_plots.png'))
    plt.close()

# ==================== MAIN FUNCTION ====================
def main():
    parser = argparse.ArgumentParser()
    
    # Model architecture
    parser.add_argument('--G', default='unet', type=str, help='unet|densenet|resnet9')
    parser.add_argument('--D', default='patch', type=str, help='patch|image')
    parser.add_argument('--norm', default='batch', type=str, help='batch|instance|none')
    parser.add_argument('--bias', default=False, type=bool)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--init_type', default='normal', type=str, help='normal|xavier|kaiming')
    
    # Training parameters
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--lambd', default=100.0, type=float, help='L1 loss weight')
    parser.add_argument('--lambd_d', default=0.5, type=float, help='D loss scale')
    parser.add_argument('--gan_loss', default='BCE', type=str, help='BCE|MSE')
    parser.add_argument('--d_update_frequency', default=1, type=int)
    parser.add_argument('--lr_decay_start', default=100, type=int, help='epoch to start lr decay')
    parser.add_argument('--lr_decay_n', default=100, type=int, help='number of epochs to decay lr to 0')
    
    # Data parameters
    parser.add_argument('--data_dir', default='./datasets/maps/', type=str)
    parser.add_argument('--unaligned', default=False, type=bool)
    parser.add_argument('--resize', default=286, type=int)
    parser.add_argument('--crop', default=256, type=int)
    
    # Output and logging
    parser.add_argument('--out_dir', default='./checkpoints', type=str)
    parser.add_argument('--pretrain_path', default='', type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--print_every_train', default=100, type=int)
    parser.add_argument('--print_every_val', default=200, type=int)
    parser.add_argument('--eval_n', default=100, type=int)
    parser.add_argument('--save_n_img', default=10000, type=int)
    parser.add_argument('--suffix', default='', type=str)
    
    # Visualization
    parser.add_argument('--vis', default=False, action='store_true')
    parser.add_argument('--port', default=8097, type=int)
    
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    
    # Print configuration
    print(f"Using {device}")
    for k, v in vars(args).items():
        print(f"{k} = {v}")
    
    # Setup output directories
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    if args.mode == "train":
        timestamp = time.strftime("%m%d%H%M%S")
        out_dir = os.path.join(args.out_dir, f"{timestamp}_{args.suffix}" if args.suffix else timestamp)
        os.makedirs(out_dir)
        out_dir_img = os.path.join(out_dir, "images")
        os.makedirs(out_dir_img)
        
        # Save config
        with open(os.path.join(out_dir, "config.txt"), "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k} = {v}\n")
    
    # Initialize model
    model = GANModel(args)
    start_epoch = 1
    
    # Load pretrained weights if specified
    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path)
        model.load_state(checkpoint['model_state'])
        if args.mode == 'train':
            start_epoch = checkpoint['epoch'] + 1
    
    model.set_start_epoch(start_epoch)
    model.to(device)
    
    # Training loop
    if args.mode == "train":
        train_loader = get_dataloader(
            os.path.join(args.data_dir, "trainA"),
            os.path.join(args.data_dir, "trainB"),
            args.batch_size, args.resize, args.crop,
            args.unaligned, device, shuffle=True, test=False
        )
        
        val_loader = get_dataloader(
            os.path.join(args.data_dir, "valA"),
            os.path.join(args.data_dir, "valB"),
            1, args.resize, args.crop,
            args.unaligned, device, shuffle=False, test=True
        )
        
        # Initialize Visdom if enabled
        if args.vis:
            viz = Visdom(port=args.port)
            # Wait for connection
            startup_sec = 5
            while not viz.check_connection() and startup_sec > 0:
                time.sleep(0.1)
                startup_sec -= 0.1
            
            if viz.check_connection():
                print("✓ Visdom connected successfully!")
                # Create windows for plots
                win_train_G = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Generator Training Loss'))
                win_train_D = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Discriminator Training Loss'))
                win_val_G = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Generator Validation Loss'))
                win_val_D = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Discriminator Validation Loss'))
            else:
                print("✗ Could not connect to Visdom! Continuing without visualization...")
                args.vis = False
        else:
            viz = None
        
        # Initialize statistics tracking
        stats = {
            'train_loss': {'G': [], 'D': [], 'G_gan': [], 'G_L1': [], 'D_real': [], 'D_fake': []},
            'val_loss': {'G': [], 'D': [], 'G_gan': [], 'G_L1': [], 'D_real': [], 'D_fake': []},
            'epochs': []
        }
        best_val_loss = float('inf')
        best_epoch = 0
        train_vis_iter = 0
        val_vis_iter = 0
        
        # Define plot function inside the training block
        def create_training_plots(stats, out_dir):
            epochs = stats['epochs']
            
            plt.figure(figsize=(15, 10))
            
            # Loss plot
            plt.subplot(2, 2, 1)
            plt.plot(epochs, stats['train_loss']['G'], 'b-', label='Train G Loss')
            plt.plot(epochs, stats['train_loss']['D'], 'b--', label='Train D Loss')
            if stats['val_loss']['G'] and len(stats['val_loss']['G']) == len(epochs):
                plt.plot(epochs, stats['val_loss']['G'], 'r-', label='Val G Loss')
                plt.plot(epochs, stats['val_loss']['D'], 'r--', label='Val D Loss')
            plt.title('Generator and Discriminator Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # GAN vs L1 Loss plot
            plt.subplot(2, 2, 2)
            plt.plot(epochs, stats['train_loss']['G_gan'], 'g-', label='Train GAN Loss')
            plt.plot(epochs, stats['train_loss']['G_L1'], 'm-', label='Train L1 Loss')
            if stats['val_loss']['G_gan'] and len(stats['val_loss']['G_gan']) == len(epochs):
                plt.plot(epochs, stats['val_loss']['G_gan'], 'g--', label='Val GAN Loss')
                plt.plot(epochs, stats['val_loss']['G_L1'], 'm--', label='Val L1 Loss')
            plt.title('GAN vs L1 Loss Components')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Discriminator Real vs Fake Loss
            plt.subplot(2, 2, 3)
            plt.plot(epochs, stats['train_loss']['D_real'], 'c-', label='Train D Real')
            plt.plot(epochs, stats['train_loss']['D_fake'], 'y-', label='Train D Fake')
            if stats['val_loss']['D_real'] and len(stats['val_loss']['D_real']) == len(epochs):
                plt.plot(epochs, stats['val_loss']['D_real'], 'c--', label='Val D Real')
                plt.plot(epochs, stats['val_loss']['D_fake'], 'y--', label='Val D Fake')
            plt.title('Discriminator Real vs Fake Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Total Loss comparison
            plt.subplot(2, 2, 4)
            train_total = [g + d for g, d in zip(stats['train_loss']['G'], stats['train_loss']['D'])]
            plt.plot(epochs, train_total, 'b-', label='Train Total Loss')
            if stats['val_loss']['G'] and stats['val_loss']['D'] and len(stats['val_loss']['G']) == len(epochs):
                val_total = [g + d for g, d in zip(stats['val_loss']['G'], stats['val_loss']['D'])]
                plt.plot(epochs, val_total, 'r-', label='Val Total Loss')
            plt.title('Total Loss (G + D)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'training_plots.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Training plots saved to {os.path.join(out_dir, 'training_plots.png')}")
        
        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            print(f"\n==== Epoch {epoch} ====")
            stats['epochs'].append(epoch)
            
            # Training
            model.G.train()
            model.D.train()
            train_losses = {k: [] for k in ['G', 'D', 'G_gan', 'G_L1', 'D_real', 'D_fake']}
            
            # Create progress bar for training
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Training Epoch {epoch}", unit="batch")
            
            for i, (x, y, idx) in pbar:
                loss = model.train((x, y, idx), save=(i == 0), out_dir_img=out_dir_img, epoch=epoch, i=i)
                
                # Update progress bar with current losses
                current_losses = {k: float(v.detach()) for k, v in loss.items()}
                pbar.set_postfix(current_losses)
                
                # Accumulate training losses
                for k, v in loss.items():
                    train_losses[k].append(float(v.detach()))
                
                # Update Visdom plots
                if args.vis and viz.check_connection():
                    viz.line(X=np.array([train_vis_iter]), Y=np.array([loss['G'].detach().cpu().numpy()]), 
                            win=win_train_G, update='append', name='Total G Loss')
                    viz.line(X=np.array([train_vis_iter]), Y=np.array([loss['G_gan'].detach().cpu().numpy()]), 
                            win=win_train_G, update='append', name='GAN Loss')
                    viz.line(X=np.array([train_vis_iter]), Y=np.array([loss['G_L1'].detach().cpu().numpy()]), 
                            win=win_train_G, update='append', name='L1 Loss')
                    viz.line(X=np.array([train_vis_iter]), Y=np.array([loss['D'].detach().cpu().numpy()]), 
                            win=win_train_D, update='append', name='Total D Loss')
                    viz.line(X=np.array([train_vis_iter]), Y=np.array([loss['D_real'].detach().cpu().numpy()]), 
                            win=win_train_D, update='append', name='D Real')
                    viz.line(X=np.array([train_vis_iter]), Y=np.array([loss['D_fake'].detach().cpu().numpy()]), 
                            win=win_train_D, update='append', name='D Fake')
                
                train_vis_iter += 1
                
                # Print detailed info less frequently
                if i % args.print_every_train == 0:
                    loss_str = " ".join([f"{k}:{v:.4f}" for k, v in loss.items()])
                    print(f"Iter {i}/{len(train_loader)} - {loss_str}")
            
            pbar.close()
            
            # Store average training losses
            for k in train_losses.keys():
                stats['train_loss'][k].append(sum(train_losses[k]) / len(train_losses[k]))
            
            # Validation
            if args.eval_n > 0:
                model.G.eval()
                model.D.eval()
                print(f"\nEvaluating {min(args.eval_n, len(val_loader))} examples...")
                
                val_losses = {k: [] for k in ['G', 'D', 'G_gan', 'G_L1', 'D_real', 'D_fake']}
                total_val_loss = 0
                val_count = 0
                
                # Create progress bar for validation
                val_pbar = tqdm(enumerate(val_loader), total=min(args.eval_n, len(val_loader)), 
                            desc="Validation", unit="batch")
                
                for i, (x, y, idx) in val_pbar:
                    if i >= args.eval_n:
                        break
                    
                    loss = model.eval((x, y, idx), save=(i == 0), out_dir_img=out_dir_img, epoch=epoch)
                    
                    # Update validation progress bar
                    current_val_losses = {k: float(v) for k, v in loss.items()}
                    val_pbar.set_postfix(current_val_losses)
                    
                    # Accumulate validation losses
                    for k, v in loss.items():
                        val_losses[k].append(float(v))
                    
                    # Update Visdom validation plots
                    if args.vis and viz.check_connection():
                        viz.line(X=np.array([val_vis_iter]), Y=np.array([loss['G']]), 
                                win=win_val_G, update='append', name='Total G Loss')
                        viz.line(X=np.array([val_vis_iter]), Y=np.array([loss['G_gan']]), 
                                win=win_val_G, update='append', name='GAN Loss')
                        viz.line(X=np.array([val_vis_iter]), Y=np.array([loss['G_L1']]), 
                                win=win_val_G, update='append', name='L1 Loss')
                        viz.line(X=np.array([val_vis_iter]), Y=np.array([loss['D']]), 
                                win=win_val_D, update='append', name='Total D Loss')
                    
                    val_vis_iter += 1
                    
                    # Calculate total validation loss for best model tracking
                    total_val_loss += loss['G'].item() + loss['D'].item()
                    val_count += 1
                
                val_pbar.close()
                
                # Store average validation losses
                for k in val_losses.keys():
                    stats['val_loss'][k].append(sum(val_losses[k]) / len(val_losses[k]))
                
                # Calculate average total validation loss (using G + D loss)
                avg_val_loss = (sum(val_losses['G']) + sum(val_losses['D'])) / val_count if val_count > 0 else float('inf')
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_model_path = os.path.join(out_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch, 
                        'val_loss': best_val_loss,
                        'model_state': model.save_state(),
                        'stats': stats
                    }, best_model_path)
                    print(f"✓ New best model saved! Epoch: {epoch}, Val Loss: {best_val_loss:.4f}")
            
            # Save stats after each epoch
            with open(os.path.join(out_dir, "stats.json"), "w") as f:
                json.dump(stats, f)
            
            model.update_scheduler()
        
        # Save final model
        final_model_path = os.path.join(out_dir, "final_model.pt")
        torch.save({
            'epoch': epoch,
            'val_loss': best_val_loss if args.eval_n > 0 else None,
            'model_state': model.save_state(),
            'stats': stats
        }, final_model_path)
        print(f"✓ Final model saved! Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
        
        # Create the plots at the end of training
        create_training_plots(stats, out_dir)

    # Testing
    elif args.mode == "test":
        test_loader = get_dataloader(
            os.path.join(args.data_dir, "testA"),
            os.path.join(args.data_dir, "testB"),
            1, args.resize, args.crop,
            args.unaligned, device, shuffle=False, test=True
        )
        
        out_dir_img = os.path.join(os.path.dirname(args.pretrain_path), "test_images")
        os.makedirs(out_dir_img, exist_ok=True)
        
        scores = {'gen': [], 'gt': []}
        model.G.eval()
        model.D.eval()
        
        for i, (x, y, idx) in enumerate(test_loader):
            if i >= args.save_n_img:
                break
            
            score_gen, score_gt = model.test((x, y, idx), i, out_dir_img)
            scores['gen'].append(float(score_gen))
            scores['gt'].append(float(score_gt))
        
        with open(os.path.join(out_dir_img, "scores.json"), "w") as f:
            json.dump(scores, f)


if __name__ == "__main__":
    main()
