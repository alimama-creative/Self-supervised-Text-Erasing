import argparse
import os
from util import util_list
import torch
from io import StringIO


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # basic parameters
        self.parser.add_argument('--online', action='store_true', help="train on dev or offline")
        self.parser.add_argument('--saveOnline', action='store_true', help="save on dev or offline")
        
        self.parser.add_argument('--dataroot', type=str, default="./examples/poster", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--checkpoints_dir', type=str, default="./checkpoints", help='where the model saved')
        self.parser.add_argument('--load_dir', default=None, help='where the model saved')
        self.parser.add_argument('--name', type=str, default="test", help='where the model saved')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        
        # model parameters
        self.parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. pix2pix')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        self.parser.add_argument('--netD', type=str, default='n_layers', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        self.parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self.parser.add_argument('--n_layers_D', type=int, default=5, help='only used if netD==n_layers')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        
        self.parser.add_argument('--adg_start', action='store_false', help='no dropout for the generator')
        self.parser.add_argument('--netD_M', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--reward_type', type=str, default="2", help='how to get the reward')
        
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--maskD', action='store_true', help='train with only hole loss')
        self.parser.add_argument('--mask_sigmoid', action='store_false', help='train with only hole loss')
        self.parser.add_argument('--PasteImage', action='store_true', help='the output are paste to the gt for loss calcu')
        self.parser.add_argument('--PasteText', action='store_true', help='the output are paste to the gt for loss calcu')
        self.parser.add_argument('--valid', type=int, default=0, help='valid for evalutions')
        self.parser.add_argument('--domain_in', action='store_true')

        # dataset process
        self.parser.add_argument('--dataset_mode', type=str, default='syn', help='chooses how datasets are loaded.')
        self.parser.add_argument('--gen_space', type=str, default='random', help='sepcific/random1/random2/random3/random4')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--crop_size', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--gen_method', type=str, default='art', help='art / basic/ copy / art_copy')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--preprocess', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        self.parser.add_argument('--flip', type=float, default=0.0, help='flip probability')
        self.parser.add_argument('--rotate', type=float, default=0.3, help='rotate probability')
        self.parser.add_argument('--mask_mode', type=int, default=1, help='the type of mask, 0 for pixel; 1 for rect')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        self.parser.add_argument('--raw_mask_dilate', type=int, default=4, help='valid for evalutions')
        self.parser.add_argument('--mask_dilate', type=int, default=3, help='valid for evalutions')

        # additional parameters
        self.parser.add_argument('--seed', type=int, default=66, help="random seed")
        self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        self.opt.data_norm = True
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        if self.isTrain:
            args = vars(self.opt)

            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')


            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.model, self.opt.name)
            util_list.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
