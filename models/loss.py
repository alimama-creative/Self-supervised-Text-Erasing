import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, is_d=None):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            if isinstance(prediction, list):
                loss = 0
                for input_i in prediction:
                    pred = input_i[-1]
                    target_tensor = self.get_target_tensor(pred, target_is_real)
                    loss += self.loss(pred, target_tensor)
            else:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'hinge':
            if is_d:
                if target_is_real:
                    prediction = -prediction
                return self.loss(1 + prediction).mean()
            else:
                return (-prediction).mean()
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, type='mixed', constant=1.0, lambda_gp=10.0, mask=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1).cuda()
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, mask)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class MaskRefineLoss(nn.Module):

    def __init__(self, type='tversky'):
        super(MaskRefineLoss, self).__init__()
        self.type = type

        if type == 'l1':
            self.criterion = torch.nn.L1Loss()

    def __call__(self, x, y):

        if self.type == 'tversky':
            # y is ground truth, x is output
            beta = 0.9
            alpha = 1 - beta
            numerator = torch.sum(x * y)
            denominator = x * y + alpha * (1 - y) * x + beta * y * (1 - x)
            return 1 - numerator / (torch.sum(denominator) + 1e-7)
        elif self.type == 'l1':
            return self.criterion(x, y)
        else:
            raise

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        # self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y, mask=None):
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        x_vgg, y_vgg = x, y

        # Compute loss
        style_loss = 0.0
        if type(mask) != type(None):
            mask1 = F.interpolate(mask, size=(256, 256), mode='nearest')
            mask2 = F.interpolate(mask, size=(128, 128), mode='nearest')
            mask3 = F.interpolate(mask, size=(64, 64), mode='nearest')
            mask4 = F.interpolate(mask, size=(32, 32), mode='nearest')
        else:
            mask1 = 1
            mask2 = 1
            mask3 = 1
            mask4 = 1

        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']*mask1), self.compute_gram(y_vgg['relu2_2']*mask1))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']*mask2), self.compute_gram(y_vgg['relu3_3']*mask2))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']*mask3), self.compute_gram(y_vgg['relu4_3']*mask3))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']*mask4), self.compute_gram(y_vgg['relu5_2']*mask4))

        return style_loss

def dice_loss(input, target):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    
    input = input 
    target = target

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss
    
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        # self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y, mask):
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        x_vgg, y_vgg = x, y

        content_loss = 0.0
        if type(mask) != type(None):
            mask1 = mask
            mask2 = F.interpolate(mask, size=(256, 256), mode='nearest')
            mask3 = F.interpolate(mask, size=(128, 128), mode='nearest')
            mask4 = F.interpolate(mask, size=(64, 64), mode='nearest')
            mask5 = F.interpolate(mask, size=(32, 32), mode='nearest')
        else:
            mask1 = 1
            mask2 = 1
            mask3 = 1
            mask4 = 1
            mask5 = 1
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1']*mask1, y_vgg['relu1_1']*mask1)
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1']*mask2, y_vgg['relu2_1']*mask2)
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1']*mask3, y_vgg['relu3_1']*mask3)
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1']*mask4, y_vgg['relu4_1']*mask4)
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1']*mask5, y_vgg['relu5_1']*mask5)


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('./pre_model/vgg19.pth'))
        features = vgg19.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1.detach(),
            'relu1_2': relu1_2.detach(),

            'relu2_1': relu2_1.detach(),
            'relu2_2': relu2_2.detach(),

            'relu3_1': relu3_1.detach(),
            'relu3_2': relu3_2.detach(),
            'relu3_3': relu3_3.detach(),
            'relu3_4': relu3_4.detach(),

            'relu4_1': relu4_1.detach(),
            'relu4_2': relu4_2.detach(),
            'relu4_3': relu4_3.detach(),
            'relu4_4': relu4_4.detach(),

            'relu5_1': relu5_1.detach(),
            'relu5_2': relu5_2.detach(),
            'relu5_3': relu5_3.detach(),
            'relu5_4': relu5_4.detach(),
        }
        return out

import os
from io import BytesIO
# from util.oss import get_bucket

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        if os.path.isfile('~/.cache/torch/hub/checkpoints/vgg16-397923af.pth'):
            vgg16 = models.vgg16(pretrained=True)
        features = vgg16.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1.detach(),
            'relu1_2': relu1_2.detach(),

            'relu2_1': relu2_1.detach(),
            'relu2_2': relu2_2.detach(),

            'relu3_1': relu3_1.detach(),
            'relu3_2': relu3_2.detach(),
            'relu3_3': relu3_3.detach(),

            'relu4_1': relu4_1.detach(),
            'relu4_2': relu4_2.detach(),
            'relu4_3': relu4_3.detach(),

            'relu5_1': relu5_1.detach(),
            'relu5_2': relu5_2.detach(),
            'relu5_3': relu5_3.detach(),
        }
        return out


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        # vgg16 = models.vgg16(pretrained=True)
        if os.path.isfile('~/.cache/torch/hub/checkpoints/vgg16-397923af.pth'):
            vgg16 = models.vgg16(pretrained=True)
        # else:
        #     vgg16 = models.vgg16(pretrained=False)
        #     path_to_load_model = "gangwei.jgw/pre_model/vgg16-397923af.pth"
        #     bucket = get_bucket(online=True)
        #     buffer = BytesIO(bucket.get_object(path_to_load_model).read())
        #     vgg16.load_state_dict(torch.load(buffer))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

