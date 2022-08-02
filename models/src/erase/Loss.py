from cv2 import PCA_DATA_AS_COL
from numpy.lib.function_base import gradient
import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F

from PIL import Image
import numpy as np
from models.loss import cal_gradient_penalty

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def visual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

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




class LossWithGAN_STE(nn.Module):
    def __init__(self, args, extractor, disc, Lamda, lr, betasInit=(0.5, 0.9)):
        super(LossWithGAN_STE, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.gan_mode = args.gan_mode
        self.discriminator = disc    ## local_global sn patch gan
        if self.gan_mode == "wgan":
            self.D_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)
        else:
            self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()
        self.lamda = Lamda
        self.paste = args.PasteImage
        self.mask_sigmoid = args.mask_sigmoid
        if self.gan_mode == 'lsgan':
            self.gloss = nn.MSELoss()
        elif self.gan_mode == 'gan':
            self.gloss = nn.BCEWithLogitsLoss()

    def forward(self, input, mask_gt, x_o1, x_o2, x_o3, output, mm, gt, raw_mask):
        self.discriminator.zero_grad()
        # requires_grad(self.discriminator, True)
        mask = mask_gt
        # mask = mm
        if self.paste:
            output = mask * input + (1 - mask) * output
        else:
            output = raw_mask * output + (1 - raw_mask) * input
        D_real = self.discriminator(gt, mask) 
        D_fake = self.discriminator(output.detach(), mask)
        # D_real = self.discriminator(gt, mask.detach()) 
        # D_fake = self.discriminator(output.detach(), mask.detach())
        Real = torch.ones_like(D_fake).cuda()
        Fake = torch.zeros_like(D_fake).cuda()
        if self.gan_mode == "vanilla":
            D_real = D_real.mean().sum() * -1
            D_real = torch.mean(F.relu(1.+D_real))
            D_fake = D_fake.mean().sum() * 1
            D_fake = torch.mean(F.relu(1.+D_fake))
        elif self.gan_mode == "gan" or self.gan_mode == "lsgan":
            D_real = torch.mean(self.gloss(D_real, Real))
            D_fake = torch.mean(self.gloss(D_fake, Fake))
        elif self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            # print(D_real.mean().shape)
            D_real = D_real.mean().sum() * -1
            D_fake = D_fake.mean().sum() * 1
        # print(D_fake.shape, D_fake)
        D_loss = D_real + D_fake  #SN-patch-GAN loss
        # D_fake = -torch.mean(D_fake)     #  SN-Patch-GAN loss
        if self.gan_mode == "wgangp":
            gradient_penalty, _ = cal_gradient_penalty(self.discriminator, gt, output.detach(), mask=mask)
            D_loss += gradient_penalty
        # if flag:
        #     print("D_real: ", torch.mean(F.relu(1.+D_real)).detach().cpu().data, "D_fake: ", torch.mean(F.relu(1.+D_fake)).detach().cpu().data)
        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        self.D_optimizer.step()

        if self.gan_mode == "wgan":
            for parm in self.discriminator.parameters():
                parm.data.clamp_(-0.01, 0.01)
        
        G_fake = self.discriminator(output, mask)
        if self.gan_mode == "vanilla":
            G_fake = G_fake.mean().sum() * -1 * 0.1
        elif self.gan_mode == "gan" or self.gan_mode == "lsgan":
            G_fake = torch.mean(self.gloss(G_fake, Real)) * 0.1
        elif self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            G_fake = -G_fake.mean().sum() * 0.1
        # requires_grad(self.discriminator, False)        
        output_comp = mask * input + (1 - mask) * output
       # import pdb;pdb.set_trace()
        holeLoss = 10 * self.l1((1 - mask) * output, (1 - mask) * gt)

        if self.paste:
            validAreaLoss = 0
            # mask_loss = dice_loss(mm * raw_mask, (1-mask) * raw_mask)
            # mask_loss = dice_loss(mm, 1-mask*raw_mask)
            mask_loss = dice_loss(mm, 1-mask_gt*raw_mask, self.mask_sigmoid)
            # mask_loss = dice_loss(mm, raw_mask-mask)
            # mask_loss = dice_loss(mm, 1-raw_mask)
        else:
            # mask_loss = dice_loss(mm, 1-mask, False)
            mask_loss = dice_loss(mm, 1-mask_gt*raw_mask, self.mask_sigmoid)
            validAreaLoss = 2*self.l1(raw_mask * mask * output, raw_mask * mask * gt)  
        ### MSR loss ###
        masks_a = F.interpolate(mask, scale_factor=0.25)
        masks_b = F.interpolate(mask, scale_factor=0.5)
        imgs1 = F.interpolate(gt, scale_factor=0.25)
        imgs2 = F.interpolate(gt, scale_factor=0.5)
        if self.paste:
            msrloss = 8 * self.l1((1-mask)*x_o3,(1-mask)*gt)+\
                6 * self.l1((1-masks_b)*x_o2,(1-masks_b)*imgs2)+\
                5 * self.l1((1-masks_a)*x_o1,(1-masks_a)*imgs1)
        else: 
            raw_masks_a = F.interpolate(raw_mask, scale_factor=0.25)
            raw_masks_b = F.interpolate(raw_mask, scale_factor=0.5)
            msrloss = 8 * self.l1((1-mask)*x_o3,(1-mask)*gt) + 0.8*self.l1(raw_mask * mask*x_o3, raw_mask * mask*gt)+\
                    6 * self.l1((1-masks_b)*x_o2,(1-masks_b)*imgs2)+1*self.l1(raw_masks_b*masks_b*x_o2,raw_masks_b*masks_b*imgs2)+\
                    5 * self.l1((1-masks_a)*x_o1,(1-masks_a)*imgs1)+0.8*self.l1(raw_masks_a*masks_a*x_o1,raw_masks_a*masks_a*imgs1)
        # msrloss = 0*msrloss
        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        GLoss = msrloss+ holeLoss + validAreaLoss+ prcLoss + styleLoss +  G_fake + 1*mask_loss
        # GLoss = holeLoss + validAreaLoss+ prcLoss + styleLoss +  G_fake + 1*mask_loss
        # if flag:
        #     print("G_GAN: %f G_L1_hole: %f G_L1_valid: %f G_prc: %f G_style: %f G_msr: %f G_mask: %f " % (0.1*D_fake, holeLoss, validAreaLoss, prcLoss, styleLoss, msrloss, mask_loss))
        return GLoss.sum(), D_real, D_fake, D_loss, G_fake, holeLoss, validAreaLoss, msrloss, prcLoss, styleLoss, mask_loss
    


