import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskInpaintGenerator(nn.Module):
    def __init__(self, in_channels=4, residual_blocks=4, use_spectral_norm=True, init_weights=True):
        super(MaskInpaintGenerator, self).__init__()
        BC = 64
        self.pre_encoder = Encoder(in_channels, BC, use_spectral_norm)
        self.encoder = Encoder(in_channels, BC, use_spectral_norm)

        MCF = 4
        self.mask_res1 = ResnetBlock(BC * MCF, 2, use_spectral_norm=use_spectral_norm)
        self.mask_res2 = ResnetBlock(BC * MCF, 2, use_spectral_norm=use_spectral_norm)
        self.mask_res3 = ResnetBlock(BC * MCF, 2, use_spectral_norm=use_spectral_norm)
        self.mask_res4 = ResnetBlock(BC * MCF, 2, use_spectral_norm=use_spectral_norm)

        ICF = 4
        self.pre_inpaint_res1 = ResnetBlock(BC * ICF, 2, use_spectral_norm=use_spectral_norm)
        self.pre_inpaint_res2 = ResnetBlock(BC * ICF, 2, use_spectral_norm=use_spectral_norm)
        self.pre_inpaint_res3 = ResnetBlock(BC * ICF, 2, use_spectral_norm=use_spectral_norm)
        self.pre_inpaint_res4 = ResnetBlock(BC * ICF, 2, use_spectral_norm=use_spectral_norm)


        self.gate1 = GateBlock(BC * MCF, BC*2, 1, use_spectral_norm=use_spectral_norm)
        self.gate2 = GateBlock(BC * MCF, BC*2, 1, use_spectral_norm=use_spectral_norm)
        self.gate3 = GateBlock(BC * MCF, BC*2, 1, use_spectral_norm=use_spectral_norm)
        self.gate4 = GateBlock(BC * MCF, BC*2, 1, use_spectral_norm=use_spectral_norm)

        blocks = []
        for _ in range(residual_blocks//2):
            block = ResnetBlock(BC * 4, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.pre_middle = nn.Sequential(*blocks)

        self.inpaint_res1 = ResnetBlock(BC * ICF, 2, use_spectral_norm=use_spectral_norm)
        self.inpaint_res2 = ResnetBlock(BC * ICF, 2, use_spectral_norm=use_spectral_norm)

        self.mask_decoder = Decoder(BC * MCF, 1, BC, use_spectral_norm)
        self.inpaint_decoder = Decoder(BC * ICF, 3, BC, use_spectral_norm)
        self.pre_inpaint_decoder = Decoder(BC * ICF, 3, BC, use_spectral_norm)


    def forward(self, x, mask_gt=None):
        image = x[:, :3, ::]
        x = self.pre_encoder(x)
        x = self.pre_middle(x)

        mx = self.mask_res1(x)
        px = self.pre_inpaint_res1(x)
        g = self.gate1(mx)
        gmx = g * px

        mx = self.mask_res2(mx)
        px = self.pre_inpaint_res2(gmx)
        g = self.gate2(mx)
        gmx = g * px

        mx = self.mask_res3(mx)
        px = self.pre_inpaint_res3(gmx)
        g = self.gate3(mx)
        gmx = g * px

        mx = self.mask_res4(mx)
        px = self.pre_inpaint_res4(gmx)
        g = self.gate4(mx)
        gmx = g * px

        mask = self.mask_decoder(mx)
        mask = torch.sigmoid(mask)

        pre_image = self.pre_inpaint_decoder(gmx)
        pre_image = (torch.tanh(pre_image) + 1) / 2

        if type(mask_gt) != type(None):
            pre_image_cmp = pre_image * mask_gt + image * (1 - mask_gt)
            x = torch.cat([pre_image_cmp, mask_gt], dim=1)
        else:
            pre_image_cmp = pre_image * mask + image * (1 - mask)
            x = torch.cat([pre_image_cmp, mask], dim=1)

        x = self.encoder(x)
        x = self.inpaint_res1(x)
        x = self.inpaint_res2(x+gmx)
        image = self.inpaint_decoder(x)
        image = (torch.tanh(image) + 1) / 2

        return image, pre_image, mask


class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        new_channel = 1
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64 + new_channel, out_channels=128, kernel_size=4,
                                    stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128 + new_channel, out_channels=256, kernel_size=4,
                                    stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256 + new_channel, out_channels=512, kernel_size=3,
                                    stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3,
                                    stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        mask = x[:, -1:, ::]
        conv1 = self.conv1(x)
        mask = F.interpolate(mask, size=(mask.size(2) // 2, mask.size(3) // 2), mode='nearest')
        conv1 = torch.cat([conv1, mask], dim=1)
        conv2 = self.conv2(conv1)
        mask = F.interpolate(mask, size=(mask.size(2) // 2, mask.size(3) // 2), mode='nearest')
        conv2 = torch.cat([conv2, mask], dim=1)
        conv3 = self.conv3(conv2)
        mask = F.interpolate(mask, size=(mask.size(2) // 2, mask.size(3) // 2), mode='nearest')
        conv3 = torch.cat([conv3, mask], dim=1)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        if self.use_sigmoid:
            conv5 = torch.sigmoid(conv5)

        return conv5, [conv1, conv2, conv3, conv4, conv5]



class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0,
                                    dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3,
                                    padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class GateBlock(nn.Module):
    def __init__(self, i_dim, base_dim, dilation=1, use_spectral_norm=False):
        super(GateBlock, self).__init__()
        BC = base_dim
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=i_dim, out_channels=BC, kernel_size=3,
                                    padding=0, dilation=dilation), use_spectral_norm),
            nn.InstanceNorm2d(BC * 2, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=BC, out_channels=1, kernel_size=3,
                                    padding=0, dilation=1), use_spectral_norm),
            nn.InstanceNorm2d(1, track_running_stats=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, in_dim, base_dim, use_spectral_norm=False):
        super(Encoder, self).__init__()
        BC = base_dim
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=BC,
                                    kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=BC, out_channels=BC * 2, kernel_size=4,
                                    stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=BC * 2, out_channels=BC * 4, kernel_size=4,
                                    stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(BC * 4, track_running_stats=False),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, base_dim, use_spectral_norm=False):
        super(Decoder, self).__init__()
        BC = base_dim
        self.conv_block = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=in_dim, out_channels=BC * 2, kernel_size=4,
                                             stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(BC * 2, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=BC * 2, out_channels=BC, kernel_size=4,
                                             stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(BC, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=BC, out_channels=out_dim, kernel_size=7, padding=0),
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
