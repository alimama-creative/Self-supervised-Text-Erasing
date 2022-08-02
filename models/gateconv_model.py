import torch
from .base_model import BaseModel
import numpy as np
import torch.nn.functional as F

import models.src.gatedconv.network as network

def on_device(online, net, gpu_ids):
    if online:
        import apex
        net = apex.parallel.convert_syncbn_model(net).cuda()
        net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)
    elif len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net


def dice_loss(input, target):
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


class GatedConvModel(BaseModel):

    def set_param(self):
        self.opt.pad_type = 'zero'
        self.opt.activation = 'elu'
        self.opt.norm = 'none'
        self.opt.in_channels = 3
        self.opt.out_channels = 3
        self.opt.cnum = 48
        self.opt.lr = 0.0001
        self.opt.dlr = 0.00001
        self.opt.beta1 = 0.5
        self.opt.beta2 = 0.9
        self.opt.dbeta1 = 0.0
        self.opt.dbeta2 = 0.9
        self.opt.gan_mode = 'hinge'
        self.opt.lambda_L1 = 10
        self.opt.init_type = "kaiming"
        self.opt.init_gain = 0.02

    def initialize(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.initialize(self, opt)
        self.set_param()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_first_L1', 'G_second_L1',  'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'comp_B', 'real_B', 'fake_B', 'mask']
        
        self.comp_flag = False
        self.mask_refine = (opt.netG=="maskrefine") 
        if opt.dataset_mode in ["items_online", "items_gen_with_noise"]:
            self.visual_names.append('comp_all')
            self.comp_flag = True
        if self.mask_refine:
            self.visual_names.append('gen_mask')
            self.loss_names.append('G_mask')
        if opt.isTrain and opt.lambda_vgg != 0:
            self.loss_names.append('G_vgg')
        print(self.visual_names)
        # self.visual_names = ['real_A', 'comp_B', 'real_B']
        # self.visual_names.append('comp_all')
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        if self.mask_refine:
            self.netG = network.GatedGeneratorWithMask(opt)
        else:
            self.netG = network.GatedGenerator(opt)
        network.weights_init(self.netG, init_type=opt.init_type, init_gain=opt.init_gain)
        self.netG = on_device(opt.online, self.netG, opt.gpu_ids)

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.netD = network.PatchDiscriminator(opt)
            network.weights_init(self.netG, init_type=opt.init_type, init_gain=opt.init_gain)
            self.netD = on_device(opt.online, self.netD, opt.gpu_ids)
            self.perceptualnet = network.PerceptualNet().cuda()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.dlr, betas=(opt.dbeta1, opt.dbeta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.raw_mask = None

        if opt.isTrain:
            self.netG.train()
        else:
            self.netG.eval()

    def set_input(self, input):
        self.real_A = input['img'].to(self.device)
        self.real_B = input['gt'].to(self.device)
        self.mask = input['mask'].to(self.device)
        if self.comp_flag:
            self.raw_mask = input['raw_mask'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.mask_refine:
            self.first_out, self.fake_B, self.gen_mask = self.netG(self.real_A)    
        else:
            self.first_out, self.fake_B = self.netG(self.real_A)  # G(A)
        # print(np.unique(np.array(self.mask.cpu())))
        self.comp_B = self.fake_B*(1-self.mask)+self.real_A*self.mask
        if self.comp_flag:
            self.comp_all = self.fake_B*(1-self.mask) + self.real_A *self.mask*self.raw_mask +self.fake_B*(1-self.raw_mask)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        fake_B = self.fake_B*(1-self.mask)+self.real_A*self.mask
    
        fake_scalar = self.netD(fake_B.detach(), self.mask)
        # True samples
        true_scalar = self.netD(self.real_B, self.mask)

        valid = torch.ones_like(fake_scalar).cuda()
        zero = torch.zeros_like(fake_scalar).cuda()

        self.loss_D_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
        self.loss_D_real = -torch.mean(torch.min(zero, -valid+true_scalar))
        # Overall Loss and optimize
        loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_B = self.fake_B*(1-self.mask)+self.real_A*self.mask
        first_out = self.first_out*(1-self.mask)+self.real_A*self.mask
        fake_scalar = self.netD(fake_B, self.mask)
        self.loss_G_GAN = -torch.mean(fake_scalar)

        self.loss_G_first_L1 = self.criterionL1(first_out, self.real_B) * self.opt.lambda_L1
        self.loss_G_second_L1 = self.criterionL1(fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_first_L1 + self.loss_G_second_L1


        if self.opt.lambda_vgg != 0:
            img_featuremaps = self.perceptualnet(self.real_B)                          # feature maps
            second_out_featuremaps = self.perceptualnet(fake_B)
            self.loss_G_vgg = self.criterionL1(second_out_featuremaps, img_featuremaps) * self.opt.lambda_vgg
            self.loss_G += self.loss_G_vgg
        # Second, G(A) = B

        if self.mask_refine:
            self.loss_G_mask = dice_loss((1-self.gen_mask)*self.raw_mask, (1-self.mask)*self.raw_mask)
            self.loss_G += self.loss_G_mask

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        # self.loss_D_real = 0
        # self.loss_D_fake = 0
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights