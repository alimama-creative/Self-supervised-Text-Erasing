import torch
import torch.distributed as dist
import numpy as np
import random
import pickle
from torchvision.models.vgg import VGG
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
import os
from torchvision import models
import torch.nn as nn
from io import BytesIO
import ntpath
from util import util_list
import pygame
pygame.init()
#VGG16 feature extract
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        # vgg16 = models.vgg16(pretrained=True)
        # print(os.path.isfile('/home/gangwei.jgw/.cache/torch/hub/checkpoints/vgg16-397923af.pth'))
        # if os.path.isfile('/home/gangwei.jgw/.cache/torch/hub/checkpoints/vgg16-397923af.pth'):
        self.vgg16 = models.vgg16(pretrained=True)
        # self.enc_1 = nn.Sequential(*self.vgg16.features[:5])
        # self.enc_2 = nn.Sequential(*self.vgg16.features[5:10])
        # self.enc_3 = nn.Sequential(*self.vgg16.features[10:17])

        # self.enc_4 = nn.Sequential(*self.vgg16.classifier[10:17])
        # fix the encoder
        # for i in range(3):
        #     for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
        #         param.requires_grad = False
        for param in self.vgg16.parameters():
            param.requires_grad = False
    def forward(self, image):
        return self.vgg16(image)


os.environ['NCCL_IB_DISABLE'] = '1'

def main():
    opt = TrainOptions().parse()

    if opt.model in ['Erase', 'erasenet', 'erase', 'gateconv']:
        opt.data_norm = False
    print(opt.data_norm)
    # initialize random seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    data_loader = CreateDataLoader(opt, dist)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0

    pre_feature = {}

    model = VGG16FeatureExtractor().to(1)
    
    # model = create_model(opt)
    # model.setup(opt, saver)
    pool = nn.MaxPool3d((256,1,1), stride=(128, 1, 1))

    for i, data in enumerate(dataset):
        # print(i)
        if i%50==0:
            print(i)
        # print(data)
        feature = model(data["gt"].to(1))
        # feature = model(data["img"].to(1))

        # feature,_,_,_,_,_,_ = model.netG(data["img"].to(model.device))
        # feature = pool(feature)
        # print(feature[0].shape)
        for j in range(len(data["path"])):
            print(data["path"][j])
            pre_feature[data["path"][j]] = feature[j].detach().cpu().numpy().flatten()
            short_path = ntpath.basename(data["path"][j])
            # print(short_path)
            # name = os.path.splitext(short_path)[0]

            # save_path = os.path.join("./visualize/img/", name+"_2.jpg")
            # print(save_path)
            # visuals = util_list.tensor2im(data["img"])
            # util_list.save_image(visuals, save_path)
            # print(pre_feature[data["path"][j]][:100])
        # if  i == 1:
        #     exit()
        # out1.write(data["path"][0]+"\n")
        total_steps += opt.batchSize
    print(total_steps)

    pre_feature_pkl = pickle.dumps(pre_feature)
    with open("./pretrain/ens_feature.pkl", "wb") as f: 
        f.write(pre_feature_pkl)

if __name__ == "__main__":
    main()