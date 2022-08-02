import torch
from .base_model import BaseModel
from . import networks
from . import loss
import numpy as np


class DiscModel(BaseModel):

    def initialize(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['D']
        # define networks (both generator and discriminator)
        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        n_layers_D=opt.n_layers_D, norm=opt.norm, add_layer=True,
                                        init_type=opt.init_type, init_gain=opt.init_gain, 
                                        gpu_ids=self.gpu_ids, online=opt.online)
        if opt.isTrain:
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.dbeta1, opt.dbeta2))
            self.optimizers.append(self.optimizer_D)

        if opt.isTrain:
            self.netD.train()
        else:
            self.netD.eval()

    def set_input(self, input):
        self.real_A = input['img'].to(self.device)
        self.real_B = input['gt'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        return

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.opt.netD == 'mask':
            pred_fake = self.netD(self.real_A, self.mask)
        else:
            pred_fake = self.netD(self.real_A)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        if self.opt.netD == 'mask':
            pred_real = self.netD(self.real_B, self.mask)
        else:
            pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def test(self):
        if self.opt.netD == 'mask':
            pred_fake = self.netD(self.real_A, self.mask)
        else:
            pred_fake = self.netD(self.real_A)
        return pred_fake.sum()


    def optimize_parameters(self):              # compute fake images: G(A)
        # update D
        # self.loss_D_real = 0
        # self.loss_D_fake = 0
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # udpate G's weights