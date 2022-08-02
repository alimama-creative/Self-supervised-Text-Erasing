from os import device_encoding
import torch
from .base_model import BaseModel
from . import networks
from . import loss
import numpy as np

class MTRNetModel(BaseModel):

    def default_param(self):
        self.opt.input_nc = 4
        self.opt.beta1 = 0.0
        self.opt.beta2 = 0.9
        self.opt.netG = 'mtr'
        self.opt.netD = 'mtr'
        self.opt.gan_mode = 'hinge'

    def initialize(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.initialize(self, opt)
        self.default_param()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_pre_GAN', 'G_pre_L1','D_pre_fake', 'MaskRefine']
        if opt.style_loss:
            self.loss_names.extend(['perceptual', 'style', 'pre_perceptual', 'pre_style'])
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'pre_fake_B', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'mtr', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.online)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc, opt.ndf, 'mtr',
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.online)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='none')
            self.criterionMask = loss.MaskRefineLoss()
            if opt.style_loss:
                self.vgg = loss.VGG19().to(self.device)
                self.criterionPerceptual = loss.PerceptualLoss()
                self.criterionStyle = loss.StyleLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*opt.d2g_lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if opt.isTrain:
            self.netG.train()
        else:
            self.netG.eval()

        self.iteration = 0

    def set_input(self, input):
        self.real_A = input['img'].to(self.device)
        self.mask_gt = input['mask'].to(self.device)
        self.mask = input['auxiliary'].to(self.device)
        self.real_B = input['gt'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.iteration += 1

        inputs = torch.cat([self.real_A, self.mask], dim=1)
        prob = np.minimum(0.9, np.ceil(self.iteration/20000)/10)
        self.use_gt_mask = False if np.random.binomial(1, prob) else True
        if self.use_gt_mask:
            self.pre_fake_B, self.fake_B, self.mask_fake = self.netG(inputs, None)  # G(A)
        else:
            self.pre_fake_B, self.fake_B, self.mask_fake = self.netG(inputs, self.mask_gt)  # G(A)
   

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.fake_B, self.mask_gt), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake, _ = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, True)
        #pre_fake_B
        pre_fake_AB = torch.cat((self.pre_fake_B, self.mask_gt), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pre_pred_fake, _ = self.netD(pre_fake_AB.detach())
        self.loss_D_pre_fake = self.criterionGAN(pre_pred_fake, False, True)
        # Real
        real_AB = torch.cat((self.real_B, self.mask_gt), 1)
        pred_real, _ = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + (self.loss_D_pre_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.fake_B, self.mask_gt), 1)
        pred_fake, _ = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, False) * self.opt.lambda_gan

        pre_fake_AB = torch.cat((self.pre_fake_B, self.mask_gt), 1)
        pre_pred_fake, _ = self.netD(pre_fake_AB)
        self.loss_G_pre_GAN = self.criterionGAN(pre_pred_fake, True, False) * self.opt.lambda_gan

        # Second, G(A) = B
        inner_weight = 3 * self.mask_gt + 0.3 * (1 - self.mask_gt)
        # print(self.criterionL1(self.fake_B, self.real_B).shape)
        self.loss_G_L1 = torch.mean(self.criterionL1(self.fake_B, self.real_B)*inner_weight)* self.opt.lambda_L1
        self.loss_G_pre_L1 = torch.mean(self.criterionL1(self.pre_fake_B, self.real_B)*inner_weight)* self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_MaskRefine = self.criterionMask(self.mask_fake, self.mask_gt) * self.opt.lambda_mask

        self.loss_G = self.loss_G_GAN + self.loss_G_pre_GAN + self.loss_G_L1 + self.loss_G_pre_L1 +self.loss_MaskRefine

        if self.opt.style_loss:
            x_vgg, y_vgg = self.vgg(self.fake_B), self.vgg(self.real_B)
            pre_x_vgg = self.vgg(self.pre_fake_B)
            # generator perceptual loss
            self.loss_perceptual = self.criterionPerceptual(x_vgg, y_vgg, inner_weight) * self.opt.lambda_content
            self.loss_pre_perceptual = self.criterionPerceptual(pre_x_vgg, y_vgg, inner_weight) * self.opt.lambda_content

            # generator style loss
            self.loss_style = self.criterionStyle(x_vgg, y_vgg, inner_weight) * self.opt.lambda_style
            self.loss_pre_style = self.criterionStyle(pre_x_vgg, y_vgg, inner_weight) * self.opt.lambda_style

            
            self.loss_G += self.loss_pre_style + self.loss_style + self.loss_pre_perceptual + self.loss_perceptual

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights