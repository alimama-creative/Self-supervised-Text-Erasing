import json
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from data.base_dataset import get_params, get_transform
from util.util_list import get_mask, gen_raw_mask, masktotensor
from util.util_list import generate_img, generate_valid_img, generate_img_with_config, random_gen_config
import os
import time
import numpy as np
import json

class ItemsDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.domain = opt.domain_in
        self.valid = (opt.valid==2 or opt.valid==3)
        self.train = (opt.isTrain and not self.valid)
        if self.train:

            self.dirs = ["train.txt"]
            # self.dirs = ["train.txt","ps_valid.txt","ps_test.txt"]
        elif opt.valid == 3: # valid image
            self.dirs = ["ps_valid.txt"]
        elif opt.valid == 2: # test image when training
            self.dirs = ["ps_test.txt"]
        elif not opt.isTrain: # image used when test
            self.dirs = ["ps_test.txt"]
        else: # easy image
            self.dirs = ["ps_valid.txt"]
        print(self.dirs)
        self.imageFiles = []
        self.labelFiles = []
        self.infos = []
        cnt = 0
        self.ps = 0
        for dir in self.dirs:
            dir = os.path.join(self.root, dir)
            with open(dir,"r") as s:
                for line in s.readlines():
                    cnt += 1
                    if cnt > self.opt.max_dataset_size and opt.isTrain and not self.valid:
                        break
                    seq = line.rstrip().split('\t')
                    self.imageFiles.append(seq[1])
                    self.infos.append(seq[2])
                    if "ps" in dir:
                        self.ps = 1
                        self.labelFiles.append(seq[3])
            if cnt > self.opt.max_dataset_size and opt.isTrain and not self.valid:
                break

        self.basic_configs = {
            "T" : 1, "F" : 10, "C" : 0, "A":0,  "S": 1, "Alpha": 0,
            "Gau": 0, "Poi": 0, "Warp": 0, "Ital": 0, "Cur": 0,
            "A": 0, "A1" : 0, "A1_S" : 0, "A1_C" : 0, 
            "A2" : 0, "A2_R" : 0, "A2_D" : 0, "A2_C" : 0,  "A2_G": 0, "A2_A": 0
        }
        self.cnt = 0
        self.backup_item = None
        self.img_size = len(self.imageFiles)
        self.mask_transform = transforms.Compose([
            transforms.Resize(size=[768, 512], interpolation=Image.BICUBIC),
            transforms.Lambda(lambda img: masktotensor(img)),
            ])
        self.input_transform = transforms.Compose([
                transforms.Resize(size=[768, 512], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ])
        self.norm_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        if not opt.isTrain:
            self.back_transform = transforms.Compose([
                transforms.Resize(size=[750, 513], interpolation=Image.BICUBIC),
            ])

    def __getitem__(self, index):
        # item = self.load_item(index)
        try:
            item = self.load_item(index)
            self.backup_item = item
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            """Handling errors introduced by random mask generation step introduced in dataloader."""
            print('loading error: item ' + self.imageFiles[index] + self.infos[index])
            if self.backup_item is not None:
                item = self.backup_item
            else:
                item = self.__getitem__(index + 1)

        return item


    def load_item(self, index):
        gt_path = os.path.join(self.root, self.imageFiles[index])
        self.cnt += 1
        gt = Image.open(gt_path).convert('RGB')

        # gt = self.input_transform(gt)
        # return {'gt': gt, 'path': gt_path}
        info = json.loads(self.infos[index])
        if self.train:
            if self.ps == 1:
                img = gt
                gt = Image.open(os.path.join(self.root, self.labelFiles[index])).convert('RGB')
            else:
                # begin = time.time()
                if "random" not in self.opt.gen_space:
                    img = generate_valid_img(gt, info, index, self.opt.gen_space)
                else:
                    self.gen_config = random_gen_config(self.opt.gen_space)
                    # img = generate_img_with_config(gt, info, self.gen_configs[np.random.randint(0,len(self.gen_configs))])
                    if self.domain:
                        img = generate_img_with_config(gt, info, self.gen_config, False)
                    else:
                        img = generate_img_with_config(gt, info, self.gen_config)
                if self.domain:
                    basic_img = generate_img_with_config(gt, info, self.basic_configs, False)
                # print(time.time()-begin)
        else:
            if self.ps == 1:
                img = gt
                gt = Image.open(os.path.join(self.root, self.labelFiles[index])).convert('RGB')
            else:
                img = generate_valid_img(gt, info, index, self.opt.gen_space)
        
        raw_mask = gen_raw_mask(img.size, info["mask"], self.opt.raw_mask_dilate)
        if self.ps == 1:
            mask = raw_mask
        else:
            mask = get_mask(gt, img, mode=self.opt.mask_mode, dilate=self.opt.mask_dilate)
        
        transforms_params = get_params(self.opt, img.size)
        trans = get_transform(self.opt, transforms_params, convert=False)
        if self.train:
            img = trans(img)
            mask = trans(mask)
            gt = trans(gt)
            raw_mask = trans(raw_mask)
            if self.domain:
                basic_img = trans(basic_img)
                basic_img = self.input_transform(basic_img)

        img = self.input_transform(img)
        gt = self.input_transform(gt)
        mask = self.mask_transform(mask)
        raw_mask = self.mask_transform(raw_mask)
        if self.opt.data_norm:
            gt = self.norm_transform(gt)
            img = self.norm_transform(img)
        if self.ps == 1:
            gt = gt*(1-mask)+img*mask
            raw_mask = np.ones_like(mask)

        if self.train and self.domain: 
            return {'img': img, 'gt': gt, 'basic_img': basic_img,'raw_mask': raw_mask, 'mask': mask, 'path': gt_path}
        return {'img': img, 'gt': gt, 'raw_mask': raw_mask, 'mask': mask, 'path': gt_path}

    def __len__(self):
        return self.img_size

    def name(self):
        return 'itemsDataset'

