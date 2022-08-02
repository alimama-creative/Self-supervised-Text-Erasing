import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

from .network_module import *

def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)

#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(nn.Module):
    def __init__(self, opt):
        super(GatedGenerator, self).__init__()
        cnum = opt.cnum
        self.vani = (opt.netG == 'vanilla')
        self.two_way = ('two' in opt.netG)
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels, cnum, 5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum, cnum * 2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 2, cnum * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 2, cnum * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # decoder
            TransposeGatedConv2d(cnum * 4, cnum * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 2, cnum * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(cnum * 2, cnum, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum, cnum//2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum//2, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, activation = 'none', norm = opt.norm),
            nn.Tanh()
        )
        if self.vani:
            self.refine_conv = nn.Sequential(
                GatedConv2d(opt.in_channels, opt.cnum, 5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum, opt.cnum, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum, opt.cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum*2, opt.cnum*2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum*2, opt.cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum*4, opt.cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum * 4, opt.cnum * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum * 4, opt.cnum * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum * 4, opt.cnum * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum * 4, opt.cnum * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
            )
            self.refine_combine = nn.Sequential(
                GatedConv2d(opt.cnum*4, opt.cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum*4, opt.cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                TransposeGatedConv2d(opt.cnum * 4, opt.cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum*2, opt.cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                TransposeGatedConv2d(opt.cnum * 2, opt.cnum, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum, opt.cnum//2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(opt.cnum//2, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, activation = 'none', norm = opt.norm),
                nn.Tanh()
            )
        else:
            self.refine_conv = nn.Sequential(
                GatedConv2d(opt.in_channels, cnum, 5, 2, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum, cnum, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum, cnum*2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*2, cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*2, cnum*4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*4, cnum*4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum * 4, cnum * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum * 4, cnum * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum * 4, cnum * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum * 4, cnum * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
            )

            self.refine_atten_1 = nn.Sequential(
                GatedConv2d(opt.in_channels, cnum, 5, 2, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum, cnum, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum, cnum*2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*2, cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*4, cnum*4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*4, cnum*4, 3, 2, 1, pad_type = opt.pad_type, activation = 'relu', norm = opt.norm)
            )
            self.refine_atten_2 = nn.Sequential(
                GatedConv2d(cnum*4, cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*4, cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
            )
            self.self_atten = Self_Attn(cnum * 4, 'relu')
    
            if self.two_way:
                cnt = 8
            else:
                cnt = 4
            self.refine_combine = nn.Sequential(
                TransposeGatedConv2d(cnum*cnt, cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*4, cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                TransposeGatedConv2d(cnum * 4, cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum*2, cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                TransposeGatedConv2d(cnum * 2, cnum, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                GatedConv2d(cnum, cnum//2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
                TransposeGatedConv2d(cnum//2, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, activation = 'none', norm = opt.norm),
                nn.Tanh()
            )


        
    def forward(self, img):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse     # in: [B, 4, H, W]
        first_out = self.coarse(img)                           # out: [B, 3, H, W]
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))
        # Refinement
        refine_conv = self.refine_conv(first_out)   

        if self.two_way:
            refine_atten = self.refine_atten_1(first_out)
            refine_atten = self.self_atten(refine_atten)
            refine_atten = self.refine_atten_2(refine_atten)
            second_out = torch.cat([refine_conv, refine_atten], dim=1)
        elif self.vani:
            second_out = refine_conv
        else:
            second_out = self.self_atten(refine_conv)
        
        second_out = self.refine_combine(second_out)
        # print(second_out.shape)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))
        return first_out, second_out


class GatedGeneratorWithMask(nn.Module):
    def __init__(self, opt):
        super(GatedGeneratorWithMask, self).__init__()
        cnum = opt.cnum
        self.coarse1 = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels, cnum, 5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum, cnum * 2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 2, cnum * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 2, cnum * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, r_mask=True),
            
        )
        self.coarse2 = nn.Sequential(
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # decoder
            TransposeGatedConv2d(cnum * 4, cnum * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum * 2, cnum * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(cnum * 2, cnum, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum, cnum//2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(cnum//2, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, activation = 'none', norm = opt.norm),
            nn.Tanh()
        ) 
        self.refine_mask = nn.Sequential(
            nn.Conv2d(cnum*4, cnum*4, 3, 1, 2),
            nn.ConvTranspose2d(cnum*4, cnum*2, 3, 1, 1),
            nn.Conv2d(cnum*2, cnum*2, 3, 1, 1),
            nn.ConvTranspose2d(cnum*2, cnum, 3, 1, 1),
            nn.Conv2d(cnum, opt.in_channels, 3, 1, 1),
            nn.Sigmoid()
        )
        self.refine_conv = nn.Sequential(
            GatedConv2d(opt.in_channels, opt.cnum, 5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum, opt.cnum, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum, opt.cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum*2, opt.cnum*2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum*2, opt.cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum*4, opt.cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum * 4, opt.cnum * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum * 4, opt.cnum * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum * 4, opt.cnum * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum * 4, opt.cnum * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(opt.cnum*4, opt.cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum*4, opt.cnum*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.cnum * 4, opt.cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum*2, opt.cnum*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.cnum * 2, opt.cnum, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum, opt.cnum//2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.cnum//2, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, activation = 'none', norm = opt.norm),
            nn.Tanh()
        )
        


        
    def forward(self, img):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse     # in: [B, 4, H, W]
        first_out, gen_mask = self.coarse1(img)                           # out: [B, 3, H, W]
        first_out = self.coarse2(first_out)
        gen_mask = self.refine_mask(gen_mask)
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))
        gen_mask = nn.functional.interpolate(gen_mask, (img.shape[2], img.shape[3]))
        # Refinement
        refine_conv = self.refine_conv(first_out)   

        second_out = refine_conv
        
        second_out = self.refine_combine(second_out)
        # print(second_out.shape)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))
        return first_out, second_out, gen_mask


#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        cnum = opt.cnum
        self.block1 = Conv2dLayer(4, cnum, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block2 = Conv2dLayer(cnum, cnum * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(cnum * 2, cnum * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(cnum * 4, cnum * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block5 = Conv2dLayer(cnum * 4, cnum * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block6 = Conv2dLayer(cnum * 4, 1, 4, 2, 1, pad_type = opt.pad_type, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask[:,:1,:,:]), 1)
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x


import os
from torchvision import models
from util.oss import get_bucket
from io import BytesIO
# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()

        if os.path.isfile('~/.cache/torch/hub/checkpoints/vgg16-397923af.pth'):
            vgg16 = models.vgg16(pretrained=True)
        else:
            vgg16 = models.vgg16(pretrained=False)
            path_to_load_model = "gangwei.jgw/pre_model/vgg16-397923af.pth"
            bucket = get_bucket(online=True)
            buffer = BytesIO(bucket.get_object(path_to_load_model).read())
            vgg16.load_state_dict(torch.load(buffer))

        block = [vgg16.features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x
