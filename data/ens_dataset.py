import json
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from data.base_dataset import get_params, get_transform
from util.util_list import get_mask, masktotensor
from util.util_list import generate_valid_img, generate_img_with_config, random_gen_config
import os
import json

class EnsDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.domain = opt.domain_in
        self.valid = (opt.valid==2 or opt.valid==3 )
        self.train = (opt.isTrain and not self.valid)
        if self.train:
            self.dirs = ["train.txt"]
            # self.dirs = ["train.txt", "train_extend.txt"]
        elif opt.valid == 3: # valid image
            self.dirs = ["valid.txt"]
        else: # test image when training
            self.dirs = ["test.txt"]

        self.imageFiles = []
        self.infos = []
        cnt = 0
        for dir in self.dirs:
            dir = os.path.join(self.root, dir)
            if self.train:
                with open(dir,"r") as s:
                    for line in s.readlines():
                        if cnt > self.opt.max_dataset_size:
                            break
                        seq = line.rstrip().split('\t')
                        if len(seq) < 3:
                            continue
                        real_path = os.path.join(self.root, seq[1])
                        self.imageFiles.append(real_path.strip())
                        self.infos.append(seq[2])
                        cnt += 1
            else:
                with open(dir,'r') as f:
                    for line in f.readlines():
                        cnt += 1
                        if cnt > self.opt.max_dataset_size:
                            break
                        real_path = os.path.join(self.root, line)
                        self.imageFiles.append(real_path.strip())
            if cnt > self.opt.max_dataset_size:
                break

        self.basic_configs = {
            "T" : 1, "F" : 10, "C" : 0, "A":0,  "S": 1, "Alpha": 0,
            "Gau": 1, "Poi": 0, "Warp": 0, "Ital": 0, "Cur": 0,
            "A": 0, "A1" : 0, "A1_S" : 0, "A1_C" : 0, 
            "A2" : 0, "A2_R" : 0, "A2_D" : 0, "A2_C" : 0,  "A2_G": 0, "A2_A": 0
        }

        self.cnt = 0
        self.backup_item = None
        self.img_size = len(self.imageFiles)
        self.mask_transform = transforms.Compose([
            transforms.Resize(size=[512, 512], interpolation=Image.BICUBIC),
            transforms.Lambda(lambda img: masktotensor(img)),
            ])
        self.input_transform = transforms.Compose([
                transforms.Resize(size=[512, 512], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ])
        self.norm_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        if not opt.isTrain:
            self.back_transform = transforms.Compose([
                transforms.Resize(size=[512, 512], interpolation=Image.BICUBIC),
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
            print('loading error: item ' + self.imageFiles[index])
            if self.backup_item is not None:
                item = self.backup_item
            else:
                item = self.__getitem__(index + 1)

        return item


    def load_item(self, index):
        gt_path = self.imageFiles[index]
        self.cnt+=1
        gt = Image.open(gt_path).convert('RGB')
        # gt = self.input_transform(gt)
        # return {"gt":gt, "path":gt_path}
        if self.train:
            info = json.loads(self.infos[index])
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
        else:
            img = gt
            gt = Image.open(gt_path.replace("all_images/", "all_labels/")).convert('RGB')
        
        # raw_mask = download_image_from_url(gt_path.replace("all_images/", "all_masks/"))
        if self.train:
            mask = get_mask(gt, img, mode=2, dilate=self.opt.mask_dilate)
            raw_mask = Image.open(gt_path.replace("all_images/", "all_masks/")).convert('RGB')
            # raw_mask = gen_raw_mask(img.size, info["mask"], self.opt.raw_mask_dilate)
        else:
            mask = Image.open(gt_path.replace("all_images/", "all_masks/")).convert('RGB')
        
        transforms_params = get_params(self.opt, img.size)
        trans = get_transform(self.opt, transforms_params, convert=False)
        if self.train:
            img = trans(img)
            mask = trans(mask)
            gt = trans(gt)
            raw_mask = trans(raw_mask)
            raw_mask = self.mask_transform(raw_mask)
            if self.domain:
                basic_img = trans(basic_img)
                basic_img = self.input_transform(basic_img)

        img = self.input_transform(img)
        gt = self.input_transform(gt)
        mask = self.mask_transform(mask)
        if self.opt.data_norm:
            gt = self.norm_transform(gt)
            img = self.norm_transform(img)
        # if self.ps == 1:
        #     gt = gt*(1-mask)+img*mask
        if not self.train:
            raw_mask = np.ones_like(mask)
        if self.train and self.domain: 
            return {'img': img, 'gt': gt, 'basic_img': basic_img,'raw_mask': raw_mask, 'mask': mask, 'path': gt_path}
        return {'img': img, 'gt': gt, 'raw_mask': raw_mask, 'mask': mask, 'path': gt_path}

    def __len__(self):
        return self.img_size

    def name(self):
        return 'EnsDataset'

