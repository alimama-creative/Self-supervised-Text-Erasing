import json
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from data.base_dataset import get_params, get_transform
from util.util_list import download_image_from_url, get_mask, gen_raw_mask, masktotensor, gen_config_from_parse
from util.util_list import generate_img, generate_valid_img, generate_img_with_config, random_gen_config
import os
import ntpath
import torch
import json
from io import BytesIO
import pickle

class EnsAdgFeaDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.domain = opt.domain_in
        self.lant = False
        self.gen_tr = False
        self.aux = opt.aux_dataset
        self.real_val = opt.real_val
        self.valid = (opt.valid==2 or opt.valid==3)
        self.train = (opt.isTrain and not self.valid and not self.aux)
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

        with open("./pretrain/ens_feature.pkl", "rb") as f:
            self.feas_dict = pickle.load(f)

        self.cnt = 0
        self.backup_item = None
        self.img_size = len(self.imageFiles)
        self.gen_configs = [random_gen_config(opt.gen_space),random_gen_config(opt.gen_space)]
        self.mask_transform = transforms.Compose([
            # transforms.Resize(size=[704, 480], interpolation=Image.BICUBIC),
            transforms.Resize(size=[512, 512], interpolation=Image.BICUBIC),
            transforms.Lambda(lambda img: masktotensor(img)),
        ])
        self.input_transform = transforms.Compose([
                # transforms.Resize(size=[704, 480], interpolation=Image.BICUBIC),
                transforms.Resize(size=[512, 512], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ])
        self.norm_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.toPIL = transforms.ToPILImage()
        if not opt.isTrain:
            self.back_transform = transforms.Compose([
                transforms.Resize(size=[512, 512], interpolation=Image.BICUBIC),
            ])
        self.controller = None

    def __getitem__(self, index):
        item = self.load_item(index)
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
        if self.train :
            # print("!!!")
            latent = self.feas_dict[gt_path]
            if self.lant:
                return {'ind': index, 'latent': latent}
        else:
            latent = []

        gt = Image.open(gt_path).convert('RGB')
        
        if self.train:
            info = json.loads(self.infos[index])
            imgs = []
            policies, _, _ = self.controller(torch.from_numpy(latent).unsqueeze(0))
            policies = policies.cpu().detach().numpy()
            config = gen_config_from_parse(policies, self.opt.gen_space)[0]
            # config = random_gen_config(self.opt.gen_space)
            imgs = [generate_img_with_config(gt, info, config, False)]
        else:
            imgs = [gt]
            gt = Image.open(gt_path.replace("all_images/", "all_labels/")).convert('RGB')
        if self.train:
            # raw_mask = gen_raw_mask(imgs[0].size, info["mask"], self.opt.raw_mask_dilate)
            raw_mask = Image.open(gt_path.replace("all_images/", "all_masks/")).convert('RGB')
            masks = []
            for im in imgs:
                masks.append(get_mask(gt, im, mode=2, dilate=self.opt.mask_dilate))
        else:
            masks = [Image.open(gt_path.replace("all_images/", "all_masks/")).convert('RGB')]

        transforms_params = get_params(self.opt, imgs[0].size)
        trans = get_transform(self.opt, transforms_params, convert=False)

        if self.train:
            gt = trans(gt)
            raw_mask = trans(raw_mask)
            raw_mask = self.mask_transform(raw_mask)
        gt = self.input_transform(gt)
        if self.opt.data_norm:
            gt = self.norm_transform(gt)

        imgs_tensor = torch.ones(len(imgs), gt.shape[0], gt.shape[1], gt.shape[2])
        masks_tensor = torch.ones(len(imgs), gt.shape[0], gt.shape[1], gt.shape[2])
        for j in range(len(imgs)):
            if self.train:
                img = trans(imgs[j])
                mask = trans(masks[j])
            else: 
                img = imgs[j]
                mask = masks[j]
            img = self.input_transform(img)
            mask = self.mask_transform(mask)
            if self.opt.data_norm:
                img = self.norm_transform(img)
            imgs_tensor[j] = img
            masks_tensor[j] = mask
        if not self.train:
            raw_mask = np.ones_like(masks_tensor[0])
        
        return {'img': imgs_tensor, 'gt': gt, 'raw_mask': raw_mask, 'mask': masks_tensor, 'path': gt_path}
        # return {'img': imgs_tensor, 'gt': gt, 'latent': latent, 'raw_mask': raw_mask, 'mask': masks_tensor, 'path': gt_path, 'policy':policies}


    def gen_data(self, index, gen_config):
        gt_path = self.imageFiles[index]
        gt = Image.open(gt_path).convert('RGB')
        info = json.loads(self.infos[index])
        imgs = []
        for config in gen_config:
            imgs.append(generate_img_with_config(gt, info, config)) 
        raw_mask = gen_raw_mask(imgs[0].size, info["mask"], self.opt.raw_mask_dilate)
        masks = []
        for im in imgs:
            masks.append(get_mask(gt, im, mode=self.opt.mask_mode, dilate=self.opt.mask_dilate))
        transforms_params = get_params(self.opt, imgs[0].size)
        trans = get_transform(self.opt, transforms_params, convert=False)
        gt = trans(gt)
        raw_mask = trans(raw_mask)
        gt = self.input_transform(gt)
        raw_mask = self.mask_transform(raw_mask)
        if self.opt.data_norm:
            gt = self.norm_transform(gt)

        imgs_tensor = torch.ones(len(imgs), gt.shape[0], gt.shape[1], gt.shape[2])
        masks_tensor = torch.ones(len(imgs), gt.shape[0], gt.shape[1], gt.shape[2])
        for j in range(len(imgs)):
            if self.train:
                img = trans(imgs[j])
                mask = trans(masks[j])
            else: 
                img = imgs[j]
                mask = masks[j]
            img = self.input_transform(img)
            mask = self.mask_transform(mask)
            if self.opt.data_norm:
                img = self.norm_transform(img)
            imgs_tensor[j] = img
            masks_tensor[j] = mask

        return {'img': imgs_tensor, 'gt': gt, 'raw_mask': raw_mask, 'mask': masks_tensor, 'path': gt_path}


    def __len__(self):
        return self.img_size

    def name(self):
        return 'itemsOnlineDataset'


