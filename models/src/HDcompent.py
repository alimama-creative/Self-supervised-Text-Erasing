import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class GatedConv2d(nn.Module):
    def __init__(
                    self, 
                    in_channels, out_channels, kernel_size,
                    stride=1, padding=0, dilation=1, groups=1, 
                    bias=True, 
                    activation=torch.nn.LeakyReLU(0.2, inplace=True)
                ):
        super(GatedConv2d, self).__init__()
        self.out_channels = out_channels
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if out_channels!=3:
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.norm = nn.InstanceNorm2d(out_channels)
            self.activation = activation
            
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)

    def forward(self, input):
        x = self.conv2d(input)

        if self.out_channels == 3 or self.activation is None:
            return x
        
        mask = self.mask_conv2d(input)

        x = x * torch.sigmoid(mask)

        if self.norm:
            x = self.norm(x)
        
        if self.activation:
            x = self.activation(x)
        
        return x


class GatedDeConv2d(nn.Module):
    def __init__(
                    self, 
                    in_channels, out_channels, kernel_size,
                    stride=1, padding=0, dilation=1, groups=1, 
                    bias=True,
                    activation=torch.nn.LeakyReLU(0.2, inplace=True)
                ):
        super(GatedDeConv2d, self).__init__()
        self.scale_factor = 2
        self.conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, activation)
    
    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor)
        return self.conv2d(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.model_type = 1
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)
       
    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if self.model_type == 1:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim),
                        activation]

        elif self.model_type==2:
            conv_block += [GatedConv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim),
                        activation]

        elif self.model_type ==3:
            conv_block += [GatedConv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim)]
        else:
            conv_block += [GatedConv2d(dim, dim, kernel_size=3, padding=p)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if self.model_type ==1:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                            norm_layer(dim)]
        elif self.model_type ==2 or self.model_type==3:
            conv_block += [GatedConv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim)]
        else:
            conv_block += [GatedConv2d(dim, dim, kernel_size=3, padding=p, activation=None)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        
        model_type = 1

        if model_type == 1:
            activation = nn.ReLU(inplace=True)
        else:
            activation = nn.LeakyReLU(0.2, inplace=True)

        ##(1):original pix2pixHD
        if model_type == 1:
            model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ##(2):replace conv with gate_conv directly
        elif model_type == 2:
            model = [nn.ReflectionPad2d(3), GatedConv2d(input_nc, ngf//2, kernel_size=7, padding=0), norm_layer(ngf//2), activation]
        ##(3):replace conv with gate_conv and remove lrrelu
        elif model_type == 3:
            model = [nn.ReflectionPad2d(3), GatedConv2d(input_nc, ngf//2, kernel_size=7, padding=0), norm_layer(ngf//2)]
        ##(4):replace conv with gate_conv and remove lrrelu and instancenorm2d
        else:
            model = [nn.ReflectionPad2d(3), GatedConv2d(input_nc, ngf//2, kernel_size=7, padding=0)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            if model_type == 1:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * mult * 2), activation]
            elif model_type == 2:
                model += [GatedConv2d(ngf * mult//2, ngf * mult, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * mult), activation]
            elif model_type == 3:
                model += [GatedConv2d(ngf * mult//2, ngf * mult, kernel_size=3, stride=2, padding=1),
                        norm_layer(ngf * mult)]
            else:
                model += [GatedConv2d(ngf * mult//2, ngf * mult, kernel_size=3, stride=2, padding=1)]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            if model_type == 1:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            else:
                model += [ResnetBlock(ngf * mult//2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            if model_type == 1:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(int(ngf * mult / 2)), activation]
            elif model_type == 2:
                model += [GatedDeConv2d(ngf * mult//2, int(ngf * mult/4), kernel_size=3, stride=1, padding=1),
                        norm_layer(int(ngf * mult /4)), activation]
            elif model_type == 3:
                model += [GatedDeConv2d(ngf * mult//2, int(ngf * mult/4), kernel_size=3, stride=1, padding=1),
                        norm_layer(int(ngf * mult /4))]
            else:
                model += [GatedDeConv2d(ngf * mult//2, int(ngf * mult/4), kernel_size=3, stride=1, padding=1)]

        if model_type == 1:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        else:
            model += [nn.ReflectionPad2d(3), GatedConv2d(ngf//2, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        
        self.model = nn.Sequential(*model)
            

    def forward(self, x):
        for i in range(len(self.model)):
            x = self.model[i](x)
            # print(i, x.shape)
            if i == 17:
                x_mask = x
            if i == 30:
                x_o1 = x
            if i == 33:
                x_o2 = x
            if i == 36:
                x_o3 = x
        return x, x_mask, x_o1, x_o2, x_o3
        # return self.model(x)      


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        model_type = 1
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            if model_type <=4:
                sequence += [[
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]
            else:
                sequence += [[
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)
                ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        if model_type <=4:
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]
        else:
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)  


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result