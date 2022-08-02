import torch
from .base_model import BaseModel
from . import networks
from . import loss
import numpy as np
import torch.nn.functional as F

class EraseModel(BaseModel):

    def set_param(self):
        self.opt.netG = 'erase'
        self.opt.netD = 'mask'
        self.opt.sigmoid = True
        self.opt.lr = 0.0001
        self.opt.dlr = 0.00001
        self.opt.beta1 = 0.5
        self.opt.beta2 = 0.9
        self.opt.dbeta1 = 0.0
        self.opt.dbeta2 = 0.9
        self.opt.gan_mode = 'hinge'
        self.opt.lambda_L1 = 10

    def initialize(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.initialize(self, opt)
        self.set_param()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_L1_hole', 'G_L1_valid', 'G_prc', 'G_style', 'G_msr', 'G_mask', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'comp_B', 'real_B', 'fake_B', 'gen_mask', 'mask']
        
        self.comp_flag = False
        if opt.dataset_mode in ["items_online"]:
            self.visual_names.append('comp_all')
            self.comp_flag = True
        self.visual_names = ['fake_B', 'comp_B', 'real_B']
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
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                         n_layers_D=opt.n_layers_D, norm=opt.norm, use_sigmoid=opt.sigmoid, 
                                         init_type=opt.init_type, init_gain=opt.init_gain, 
                                         gpu_ids=self.gpu_ids, online=opt.online)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.vgg = loss.VGG16FeatureExtractor().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.dlr, betas=(opt.dbeta1, opt.dbeta2))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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
        self.x_o1, self.x_o2, self.x_o3, self.fake_B, self.gen_mask = self.netG(self.real_A)  # G(A)
        # print(np.unique(np.array(self.mask.cpu())))
        self.comp_B = self.fake_B*(1-self.mask)+self.real_A*self.mask
        if self.comp_flag:
            self.comp_all = self.fake_B*(1-self.mask) + self.real_A *self.mask*self.raw_mask +self.fake_B*(1-self.raw_mask)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.opt.PasteImage:
            fake_AB = self.fake_B*(1-self.mask)+self.real_A*self.mask
        elif self.opt.PasteText:
            fake_AB = self.fake_B*self.raw_mask+self.real_A*(1-self.raw_mask)
        else:
            fake_AB = self.fake_B # we use conditional GANs; we need to feed both input and output to the discriminator
        if self.opt.netD == 'mask':
            pred_fake = self.netD(fake_AB.detach(), self.mask)
        else:
            pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, True)
        # Real
        real_AB = self.real_B
        if self.opt.netD == 'mask':
            pred_real = self.netD(real_AB, self.mask)
        else:
            pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 1.0
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.PasteImage:
            fake_AB = self.fake_B*(1-self.mask)+self.real_A*self.mask
        elif self.opt.PasteText:
            fake_AB = self.fake_B*self.raw_mask+self.real_A*(1-self.raw_mask)
        else:
            fake_AB = self.fake_B
        if self.opt.netD == 'mask':
            pred_fake = self.netD(fake_AB, self.mask)
        else:
            pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, False) * 0.1
        # self.loss_G_GAN = 0
        # Second, G(A) = B
        self.loss_G_L1_hole = self.criterionL1(self.fake_B*(1-self.mask), self.real_B*(1-self.mask)) * 10

        if self.opt.PasteImage:
            self.loss_G_mask = loss.dice_loss(self.gen_mask*self.raw_mask, (1-self.mask)*self.raw_mask)
            self.loss_G_L1_valid = 0
        elif self.opt.PasteText:
            self.loss_G_mask = loss.dice_loss(self.gen_mask*self.raw_mask, (1-self.mask)*self.raw_mask)
            self.loss_G_L1_valid = 2*self.criterionL1(self.fake_B*self.mask*self.raw_mask, self.real_B*self.mask*self.raw_mask)  
        else:
            self.loss_G_mask = loss.dice_loss(self.gen_mask, 1-self.mask)
            self.loss_G_L1_valid = 2*self.criterionL1(self.fake_B*self.mask, self.real_B*self.mask)  
        
        ### MSR loss ###
        masks_a = F.interpolate(self.mask, scale_factor=0.25)
        masks_b = F.interpolate(self.mask, scale_factor=0.5)
        imgs1 = F.interpolate(self.real_B, scale_factor=0.25)
        imgs2 = F.interpolate(self.real_B, scale_factor=0.5)
        if self.opt.PasteImage:
            self.loss_G_msr = 8 * self.criterionL1((1-self.mask)*self.x_o3,(1-self.mask)*self.real_B)+\
                6 * self.criterionL1((1-masks_b)*self.x_o2,(1-masks_b)*imgs2)+\
                5 * self.criterionL1((1-masks_a)*self.x_o1,(1-masks_a)*imgs1)
        elif self.opt.PasteText:
            raw_masks_a = F.interpolate(self.raw_mask, scale_factor=0.25)
            raw_masks_b = F.interpolate(self.raw_mask, scale_factor=0.5)
            self.loss_G_msr = self.criterionL1((1-self.mask)*self.x_o3, (1-self.mask)*self.real_B) +\
                    0.8*self.criterionL1(self.mask*self.x_o3*self.raw_mask, self.mask*self.real_B*self.raw_mask)+\
                    6 * self.criterionL1((1-masks_b)*self.x_o2, (1-masks_b)*imgs2)+\
                    1*self.criterionL1(masks_b*self.x_o2*raw_masks_b, masks_b*imgs2*raw_masks_b)+\
                    5 * self.criterionL1((1-masks_a)*self.x_o1, (1-masks_a)*imgs1)+\
                    0.8*self.criterionL1(masks_a*self.x_o1*raw_masks_a, masks_a*imgs1*raw_masks_a)
        else:
            self.loss_G_msr = self.criterionL1((1-self.mask)*self.x_o3, (1-self.mask)*self.real_B) +\
                    0.8*self.criterionL1(self.mask*self.x_o3, self.mask*self.real_B)+\
                    6 * self.criterionL1((1-masks_b)*self.x_o2, (1-masks_b)*imgs2)+\
                    1*self.criterionL1(masks_b*self.x_o2, masks_b*imgs2)+\
                    5 * self.criterionL1((1-masks_a)*self.x_o1, (1-masks_a)*imgs1)+\
                    0.8*self.criterionL1(masks_a*self.x_o1, masks_a*imgs1)

        out_comp = self.fake_B*(1-self.mask)+self.real_A*self.mask
        feat_output_comp = self.vgg(out_comp)
        feat_output = self.vgg(fake_AB)
        feat_gt = self.vgg(self.real_A)

        self.loss_G_prc = 0.0
        for i in range(3):
            self.loss_G_prc += 0.01 * self.criterionL1(feat_output[i], feat_gt[i])
            self.loss_G_prc += 0.01 * self.criterionL1(feat_output_comp[i], feat_gt[i])

        self.loss_G_style = 0.0
        for i in range(3):
            self.loss_G_style += 120 * self.criterionL1(loss.gram_matrix(feat_output[i]),
                                          loss.gram_matrix(feat_gt[i]))
            self.loss_G_style += 120 * self.criterionL1(loss.gram_matrix(feat_output_comp[i]),
                                          loss.gram_matrix(feat_gt[i]))


        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1_hole + self.loss_G_L1_valid +\
                      self.loss_G_mask + self.loss_G_prc + self.loss_G_style + self.loss_G_msr
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