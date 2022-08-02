from cv2 import FastFeatureDetector
import torch
from .base_model import BaseModel
from . import networks
from . import loss
import numpy as np
import torch.nn.functional as F
from models.src.erase.Loss import LossWithGAN_STE
from models.src.erase.Model import VGG16FeatureExtractor
from models.src.erase.sa_gan import STRnet2, Dis_Mask, init_weights
from models.src.erase.discriminator import Discriminator_STE
import time
from torch import autograd

def svd(L):
    try:
        u, s, v = torch.svd(L)
    except:                     # torch.svd may have convergence issues for GPU and CPU.
        u, s, v = torch.svd(L + 1e-4*L.mean()*torch.rand(L.shape[0], L.shape[1]).cuda())
    return u,s,v

def RSD(Feature_s, Feature_t):
    u_s, s_s, v_s = svd(Feature_s.t())
    u_t, s_t, v_t = svd(Feature_t.t())
    p_s, cospa, p_t = svd(torch.mm(u_s.t(), u_t))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return torch.norm(sinpa,1)+0.01*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)


class AdaptiveFeatureNorm(torch.nn.Module):
    def __init__(self, delta):
        super(AdaptiveFeatureNorm, self).__init__()
        self.delta = delta

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        radius = f.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + self.delta
        loss = ((f.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss


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

    return 1 - d

def on_device(online, net, gpu_ids):
    if online:
        import apex
        net = apex.parallel.convert_syncbn_model(net).cuda()
        net = torch.nn.parallel.DistributedDataParallel(net,find_unused_parameters=True)
    elif len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net

class EraseModel(BaseModel):

    def set_param(self):
        self.opt.netG = 'erase'
        self.opt.netD = 'mask'
        # self.opt.lr = 0.0001
        # self.opt.dlr = 0.00001
        self.opt.beta1 = 0.5
        self.opt.beta2 = 0.9
        self.opt.dbeta1 = 0.0
        self.opt.dbeta2 = 0.9
        # self.opt.gan_mode = 'hinge'
        self.opt.lambda_L1 = 10

    def initialize(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.initialize(self, opt)
        self.set_param()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_L1_hole', 'G_L1_valid', 'G_prc', 'G_style', 'G_msr', 'G_mask','D_real', 'D_fake', "D"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'comp_B', 'real_B', 'fake_B', 'gen_mask', 'mask']
        self.visual_names = ['real_A', 'comp_B', 'real_B', 'fake_B', 'x_o3', 'gen_mask', 'mask', 'comp_G']
        self.comp_flag = False

        self.visual_names.append('comp_all')
        self.visual_names.append('mask_all')
        self.comp_flag = True
    
        self.adaptive_feature_norm = AdaptiveFeatureNorm(1.0).to(self.device)
        # self.visual_names = ['real_A', 'comp_B', 'real_B']
        # self.visual_names.append('comp_all')
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:        
            self.continual_loss = self.opt.continual_loss
            if self.continual_loss:
                self.loss_names.append('G_conti')
                self.loss_G_conti = 0
                self.continual_loss = False
            
            self.contrastive_loss = self.opt.contrastive_loss
            if self.contrastive_loss:
                self.loss_names.append('G_contr')
                self.loss_G_contr = 0
                # self.contrastive_loss = False
            self.model_names = ['G', 'Criterion']
            # self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        self.netG = STRnet2(3)
        init_weights(self.netG, init_type='normal', init_gain=0.02)
        self.netG = on_device(opt.online, self.netG, opt.gpu_ids)

        self.mask_loss_type = False
        if "4" in opt.reward_type:
            self.mask_loss_type = True
        self.D_M = opt.netD_M
        if self.isTrain:
            if self.D_M or self.opt.baseline == "dann":
                # self.loss_names.append('D_mask')
                self.model_names.append('D_M')
                self.netD_M = Dis_Mask(3)
                init_weights(self.netD_M, init_type='normal', init_gain=0.02)
                self.netD_M = on_device(opt.online, self.netD_M, opt.gpu_ids)
                # self.optimizer_D_m = torch.optim.Adam(self.netD_M.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                if self.opt.baseline == "dann":
                    self.optimizer_D_m = torch.optim.Adam(self.netD_M.parameters(), lr=opt.lr/2.0, betas=(opt.beta1, opt.beta2))
                else:
                    self.optimizer_D_m = torch.optim.Adam(self.netD_M.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                
                # self.optimizer_D_m = torch.optim.Adam(self.netD_M.parameters(), lr=opt.dlr, betas=(opt.beta1, opt.beta2))
            self.netD = Discriminator_STE(3, opt.sigmoid)
            self.netCriterion = LossWithGAN_STE(opt, VGG16FeatureExtractor(), self.netD, lr=opt.dlr, betasInit=(0.0, 0.9), Lamda=10.0)   
            init_weights(self.netCriterion.discriminator, init_type='normal', init_gain=0.02)
            self.netCriterion = on_device(opt.online, self.netCriterion, opt.gpu_ids)
            self.dis = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if self.opt.gan_mode == "wgan":
                self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.netCriterion.module.D_optimizer)

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
            self.mask_all = 1 - self.mask * self.raw_mask
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
        if self.comp_flag:
            # print(self.mask.shape, self.raw_mask.shape)
            self.mask_all = 1 - self.mask * self.raw_mask
        if "pub" in self.opt.dataset_mode:
            self.real_B = self.real_Bs[:,no,:]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.x_o1, self.x_o2, self.x_o3, self.fake_B, self.gen_mask = self.netG(self.real_A)
        # print(self.real_A.shape)
        self.x_o0, self.x_o1, self.x_o2, self.x_o3, self.fake_B, self.gen_mask, self.x_mask = self.netG(self.real_A)  # G(A)
        # self.fake_B = self.x_o3
        # print(np.unique(np.array(self.mask.cpu())))
        self.comp_B = self.fake_B*(1-self.mask)+self.real_A*self.mask

        if self.opt.mask_sigmoid:
            threshhold = 0.5
            gen_mask = self.gen_mask.clone().detach()
            ones = gen_mask >= threshhold
            zeros = gen_mask < threshhold
            gen_mask.masked_fill_(ones, 1.0)
            gen_mask.masked_fill_(zeros, 0.0)
            mask = 1-gen_mask
        else:
            mask = 1-self.gen_mask
        # print(self.opt.mask_sigmoid)
        self.comp_G = self.fake_B*(1-mask)+self.real_A*mask

        if self.comp_flag:
            self.comp_all = self.fake_B*(1-self.mask) + self.real_A *self.mask*self.raw_mask +self.fake_B*(1-self.raw_mask)

    def forward_basic(self):
        _, self.basic_x_o1, self.basic_x_o2, self.basic_x_o3, self.basci_fake, _, _ = self.netG(self.basic)  # G(A)
        

    def optimize_parameters(self, stype="n"):
        self.forward()                   # compute fake images: G(A)
        # update D
        if stype == "dann" or (self.D_M and not self.mask_loss_type) :
            self.optimize_mask_dis()
        
        G_loss, self.loss_D_real, self.loss_D_fake, self.loss_D, self.loss_G_GAN, self.loss_G_L1_hole, \
                self.loss_G_L1_valid, self.loss_G_msr, self.loss_G_prc, self.loss_G_style, self.loss_G_mask= \
                self.netCriterion(self.real_A, self.mask, self.x_o1,\
                                self.x_o2, self.x_o3, self.fake_B, \
                                self.gen_mask, self.real_B, self.raw_mask)
        if self.continual_loss:
            self.loss_G_conti = self._compute_consolidation_loss(self.opt.cont_weight*100000)
            # print(G_loss, self.loss_G_conti)
            G_loss += self.loss_G_conti
        
        if self.contrastive_loss:
            self.coarse_comp_B = self.x_o3.detach()*(1-self.mask)+self.real_A*self.mask
            print(self.dis(self.comp_B, self.real_A), self.dis(self.coarse_comp_B, self.comp_B))
            # self.loss_G_contr = 5. * self.dis(self.comp_B, self.real_A) / (1e-7 + self.dis(self.comp_B, self.real_A) + 10. * self.dis(self.coarse_comp_B, self.comp_B))
            # self.loss_G_contr = 0.5 * self.dis(self.comp_B, self.real_A) / (1e-7 + self.dis(self.comp_B, self.real_A) + 10. * self.dis(self.coarse_comp_B, self.comp_B))
            lam, pro, ex = 0.01, 50.0, 2.0
            self.loss_G_contr = lam * self.dis(self.comp_B, self.real_A) / (1e-7 + self.dis(self.comp_B, self.real_A) + pro * max(1e-7, self.dis(self.coarse_comp_B, self.comp_B)**ex))
            # print(G_loss, self.loss_G_conti)
            G_loss += self.loss_G_contr
        if stype == "focal":
            alpha = min(1, ((self.loss_G_L1_hole.detach())*12)**2)
            # alpha = min(1, ((G_loss.detach())*10)**2)
            # print("alpha:", alpha)
            G_loss = alpha * G_loss.sum()
        elif stype == "domain":
            if "domain" not in self.loss_names:
                self.loss_names.append("domain")

            if "basic" not in self.visual_names:
                self.visual_names.append("basic")
            # self.loss_domain = 8 * self.dis(self.basic_x_o1.detach(), self.x_o1) + \
            #                6 * self.dis(self.basic_x_o2.detach(), self.x_o2) + \
            #                5 * self.dis(self.basic_x_o3.detach(), self.x_o3)
            mask_a = F.interpolate(self.mask, scale_factor=0.25)
            mask_b = F.interpolate(self.mask, scale_factor=0.5)
            self.loss_domain = 8 * self.dis((1-self.mask)*self.basic_x_o3.detach(), (1-self.mask)*self.x_o3) + \
                           6 * self.dis((1-mask_b)*self.basic_x_o2.detach(), (1-mask_b)*self.x_o2) + \
                           5 * self.dis((1-mask_a)*self.basic_x_o1.detach(), (1-mask_a)*self.x_o1)
            G_loss = G_loss.sum() + self.loss_domain
            # print(self.loss_domain)
        elif stype == "rsd":
            if "rsd" not in self.loss_names:
                self.loss_names.append("rsd")
            m = torch.nn.MaxPool2d((24, 16), stride=(2, 1))
            self.target_x_o0, _, _, _, _, _, _ = self.netG(self.real_A)
            self.loss_rsd = 0.01 * RSD(m(self.x_o0).view(self.x_o0.size(0),-1), m(self.target_x_o0).view(self.x_o0.size(0),-1))
            G_loss = G_loss.sum() + self.loss_rsd
        elif stype == "afn":
            if "afn" not in self.loss_names:
                self.loss_names.append("afn")
            m = torch.nn.MaxPool2d((24, 16), stride=(2, 1))
            # m = torch.nn.MaxPool2d((16, 16), stride=(2, 1))
            self.target_x_o0, _, _, _, _, _, _ = self.netG(self.real_A)
            self.loss_afn =  0.05 * (self.adaptive_feature_norm(m(self.x_o0).view(self.x_o0.size(0),-1)) +self.adaptive_feature_norm(m(self.target_x_o0).view(self.x_o0.size(0),-1)))
            G_loss = G_loss.sum() + self.loss_afn
        elif stype == "dann":
            if "dann" not in self.loss_names:
                self.loss_names.append("dann")            
            d_mm = self.netD_M(self.x_mask)
            self.loss_dann = torch.mean(dice_loss(d_mm, 1-self.mask*self.raw_mask, sig=False))
            G_loss = G_loss.sum() + self.loss_dann
        else:
            G_loss = G_loss.sum()
        self.loss_G = G_loss
        self.optimizer_G.zero_grad()
        G_loss.backward()
        self.optimizer_G.step()              # udpate G's weights

        # 'G', 'G_GAN', 'G_L1_hole', 'G_L1_valid', 'G_prc', 'G_style', 'G_msr', 'G_mask', 'D_real', 'D_fake', 'D'

    def optimize_mask_dis(self):
        D_mask = self.netD_M(self.x_mask.detach())
        self.loss_D_mask = torch.mean(dice_loss(D_mask, 1-self.raw_mask, sig=False))
        # if self.mask_loss_type:
        #     # self.loss_D_mask = dice_loss(D_mask*self.mask, 1-self.raw_mask, sig=False)
        #     self.loss_D_mask = torch.mean(dice_loss(D_mask, 1-self.raw_mask, sig=False))
        # else:
        #     self.loss_D_mask = torch.mean(dice_loss(D_mask, self.mask-self.raw_mask, sig=False))
        self.optimizer_D_m.zero_grad()
        self.loss_D_mask.backward()
        self.optimizer_D_m.step()
    
    def update_val_real_dis(self, real_dataset, gen_configs):
        real_dataset.dataset.gen_tr = True
        real_dataset.dataset.gen_configs = gen_configs
        for _ in range(1):
            for data in real_dataset:
                self.set_inputs(data)
                self.set_specific_image(0)
                self.forward()
                self.optimize_mask_dis()


    def calcu_loss(self):
        # G_loss, _, _, _, _, _, _, _, _, _, _= \
        #         self.netCriterion(self.real_A, self.mask, self.x_o1,\
        #                         self.x_o2, self.x_o3, self.fake_B, \
        #                         self.gen_mask, self.real_B, self.raw_mask)
        # G_loss = self.dis(self.comp_B, self.real_B)
        G_loss = F.l1_loss(self.comp_B, self.real_B, reduction='none').mean([1,2,3])
        # # G_loss = G_loss.sum()
        # print(G_loss.shape)
        return G_loss

    def calcu_real_reward(self):
        D_mask = self.netD_M(self.x_mask.detach())
        real_reward = dice_loss(D_mask*self.raw_mask, 1-self.mask, sig=False)
        # if self.mask_loss_type:
        #     real_reward = dice_loss(D_mask*self.raw_mask, 1-self.mask, sig=False)
        # else:
        #     real_reward = dice_loss(D_mask, 1-self.mask*self.raw_mask, sig=False)
        return real_reward * D_mask.shape[0]

    def _update_mean_params(self):
        for param_name, param in self.netG.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.netG.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, dataset, num_batch, dist):
        print("update fisher params")
        begin = time.time()
        for i, data in enumerate(dataset):
            self.set_inputs(data)
            self.set_specific_image(0)
            self.forward()
            G_loss = self.dis(self.comp_B, self.real_B)
            if i==0:    
                grad_sum = autograd.grad(G_loss, self.netG.parameters(), allow_unused=True)
            else:
                curr_grad = autograd.grad(G_loss, self.netG.parameters(), allow_unused=True)
                for curr_para, para in zip(curr_grad, grad_sum):
                    if para is None:
                        # print(para)
                        continue
                    para.data += curr_para.data
                    if i == len(dataset)-1:
                        para.data /= len(dataset)

        print(time.time()-begin)
        _buff_param_names = [param[0].replace('.', '__') for param in self.netG.named_parameters()]
    
        for _buff_param_name, param in zip(_buff_param_names, grad_sum):
            if param is None:
                continue
            self.netG.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)
    
    def register_ewc_params(self, dataset, num_batch, dist=None):
        # if dist.get_rank() == 0:
        self._update_fisher_params(dataset, num_batch, dist)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        losses = []
        for param_name, param in self.netG.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            try:
                estimated_fisher = getattr(self.netG, '{}_estimated_fisher'.format(_buff_param_name))
                estimated_mean = getattr(self.netG, '{}_estimated_mean'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            except AttributeError:
                continue
        return (weight / 2) * sum(losses)