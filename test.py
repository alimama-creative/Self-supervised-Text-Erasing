import os
import torch
from options.test_options import TestOptions
import torch.distributed as dist
from data import CreateDataLoader
from models import create_model
from util import html_util as html
from util.visualizer import save_images
import pygame
import ntpath
import numpy as np
import random

os.environ['NCCL_IB_DISABLE'] = '1'

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
if opt.model in ['Erase','erase','erasenet', 'gateconv']:
    opt.data_norm = False

# opt.isTrain = True
data_loader = CreateDataLoader(opt, dist)
dataset = data_loader.load_data()

# opt.isTrain = False
model = create_model(opt)
pygame.init()

model.setup(opt)

back_trans = None

if hasattr(data_loader.dataset, "back_transform"):
    back_trans = data_loader.dataset.back_transform
    print("reshape trans")

web_dir = os.path.join("../ablation/", opt.model, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))
for i, data in enumerate(dataset):
    if i > opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals(back_trans)
    img_path = model.get_image_paths()
    if i % 10 == 0:
        print('process image...%s, %s' % (i, img_path))
    save_images(webpage, visuals, img_path)

webpage.save()


# python test.py --dataset_mode syn_online --model pix2pix --name unet --netG unet_512  --n_layers_D 6 --which_epoch latest


