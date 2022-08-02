import torch
from .base_model import BaseModel
from . import networks
from . import loss
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(input, target, sig=True):
    if sig:
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


class Pix2PixModel(BaseModel):

    def initialize(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'comp_B', 'real_B', 'fake_B', ]
        self.comp_flag = False
        if opt.dataset_mode in ["items_adg", "items_online", "items_gen_with_noise"]:
            self.visual_names.append('comp_all')
            self.comp_flag = True
        # self.visual_names = ['real_A', 'comp_B', 'real_B',]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                    norm=opt.norm, use_dropout=not opt.no_dropout, 
                                    init_type=opt.init_type, init_gain=opt.init_gain, 
                                    gpu_ids=self.gpu_ids, online=opt.online)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                         n_layers_D=opt.n_layers_D, norm=opt.norm, 
                                         init_type=opt.init_type, init_gain=opt.init_gain, 
                                         gpu_ids=self.gpu_ids, online=opt.online)
        self.dis = torch.nn.L1Loss()
        self.mask_loss_type = False
        if "4" in opt.reward_type:
            self.mask_loss_type = True
        self.D_M = opt.netD_M
        if self.isTrain:
            # define loss functions
            if self.D_M:
                # self.loss_names.append('D_mask')
                self.model_names.append('D_M')
                self.netD_M = networks.define_D_M(init_gain=opt.init_gain, 
                                         gpu_ids=self.gpu_ids, online=opt.online)
                self.optimizer_D_m = torch.optim.Adam(self.netD_M.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.dlr, betas=(opt.dbeta1, opt.dbeta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if opt.isTrain:
            self.netG.train()
        else:
            self.netG.eval()

        self.conv_o1 = nn.Conv2d(128,3,kernel_size=1).cuda()
        self.conv_o2 = nn.Conv2d(64,3,kernel_size=1).cuda()
        self.conv_o3 = nn.Conv2d(32,3,kernel_size=1).cuda()

    def set_input(self, input):
        self.real_A = input['img'].to(self.device)
        self.real_B = input['gt'].to(self.device)
        self.mask = input['mask'].to(self.device)
        if self.comp_flag:
            self.raw_mask = input['raw_mask'].to(self.device)
        self.image_paths = input['path']

    def set_inputs(self, input):
        self.real_As = input['img'].to(self.device)
        if "pub" in self.opt.dataset_mode:
            self.real_Bs = input['gt'].to(self.device)
        else:
            self.real_B = input['gt'].to(self.device)
        self.masks = input['mask'].to(self.device)
        if self.comp_flag:
            self.raw_mask = input['raw_mask'].to(self.device)
        self.image_paths = input['path']
    
    def set_basic_input(self, input):
        self.basic = input['basic_img'].to(self.device)

    def set_specific_image(self, no):
        self.real_A = self.real_As[:,no,:]
        self.mask = self.masks[:,no,:]
        if "pub" in self.opt.dataset_mode:
            self.real_B = self.real_Bs[:,no,:]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.x_mask, self.x_o1, self.x_o2, self.x_o3 = self.netG(self.real_A)  # G(A)
        # print(np.unique(np.array(self.mask.cpu())))
        self.comp_B = self.fake_B*(1-self.mask)+self.real_A*self.mask
        if self.comp_flag:
            self.comp_all = self.fake_B*(1-self.mask) + self.real_A *self.mask*self.raw_mask +self.fake_B*(1-self.raw_mask)

    def forward_basic(self):
        _, _, self.basic_x_o1, self.basic_x_o2, self.basic_x_o3 = self.netG(self.basic)  # G(A)
        

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.opt.PasteImage:
            fake_AB = torch.cat((self.real_A, self.fake_B*(1-self.mask)+self.real_A*self.mask), 1)
        # elif self.opt.PasteText:
        #     fake_AB = torch.cat((self.real_A, self.fake_B*self.raw_mask+self.real_A*(1-self.raw_mask)), 1)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        if self.opt.netD == 'mask':
            pred_fake = self.netD(fake_AB.detach(), self.mask)
        else:
            pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        if self.opt.netD == 'mask':
            pred_real = self.netD(real_AB, self.mask)
        else:
            pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self, stype):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.PasteImage:
            comp_out = self.fake_B*(1-self.mask)+self.real_A*self.mask       
        # elif self.opt.PasteText:
        #     comp_out = self.fake_B*self.raw_mask+self.real_A*(1-self.raw_mask)
        else:
            comp_out = self.fake_B
        fake_AB = torch.cat((self.real_A, comp_out), 1) 
        if self.opt.netD == 'mask':
            pred_fake = self.netD(fake_AB, self.mask)
        else:
            pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(comp_out, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        if stype == "focal":
            alpha = min(1, ((self.loss_G_L1.detach())*15)**2)
            # alpha = min(1, ((self.loss_G.detach())*10)**2)
            # print("alpha:", alpha)
            self.loss_G = alpha * self.loss_G
        elif stype == "domain":
            if "domain" not in self.loss_names:
                self.loss_names.append("domain")
            # self.loss_domain = 8 * self.dis(self.basic_x_o1.detach(), self.x_o1) + \
            #                6 * self.dis(self.basic_x_o2.detach(), self.x_o2) + \
            #                5 * self.dis(self.basic_x_o3.detach(), self.x_o3)
            mask_a = F.interpolate(self.mask, scale_factor=0.25)
            mask_b = F.interpolate(self.mask, scale_factor=0.5)
            self.loss_domain = 8 * self.dis((1-self.mask)*self.conv_o3(self.basic_x_o3.detach()), (1-self.mask)*self.conv_o3(self.x_o3)) + \
                           6 * self.dis((1-mask_b)*self.conv_o2(self.basic_x_o2.detach()), (1-mask_b)*self.conv_o2(self.x_o2)) + \
                           5 * self.dis((1-mask_a)*self.conv_o1(self.basic_x_o1.detach()), (1-mask_a)*self.conv_o1(self.x_o1))
            self.loss_G = self.loss_G + self.loss_domain
            # print(self.loss_domain)
        else:
            self.loss_G = self.loss_G
        self.loss_G.backward()

    def optimize_parameters(self, stype="n"):
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
        self.backward_G(stype)                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

        if self.D_M and not self.mask_loss_type:
            self.optimize_mask_dis()

    def optimize_mask_dis(self):
        D_mask = self.netD_M(self.x_mask.detach())
        if self.mask_loss_type:
            # self.loss_D_mask = dice_loss(D_mask*self.mask, 1-self.raw_mask, sig=False)
            self.loss_D_mask = dice_loss(D_mask, 1-self.raw_mask, sig=False)
        else:
            self.loss_D_mask = dice_loss(D_mask, self.mask-self.raw_mask, sig=False)
        self.optimizer_D_m.zero_grad()
        self.loss_D_mask.backward()
        self.optimizer_D_m.step()
    
    def calcu_loss(self):
        # G_loss, _, _, _, _, _, _, _, _, _, _= \
        #         self.netCriterion(self.real_A, self.mask, self.x_o1,\
        #                         self.x_o2, self.x_o3, self.fake_B, \
        #                         self.gen_mask, self.real_B, self.raw_mask)
        G_loss = self.dis(self.comp_B, self.real_B)
        G_loss = G_loss.sum()
        return G_loss

    def calcu_real_reward(self):
        D_mask = self.netD_M(self.x_mask.detach())
        if self.mask_loss_type:
            real_reward = dice_loss(D_mask*self.raw_mask, 1-self.mask, sig=False)
        else:
            real_reward = dice_loss(D_mask, 1-self.mask*self.raw_mask, sig=False)
        # print(D_mask.shape[0])
        # print(real_reward)
        return real_reward * D_mask.shape[0]
