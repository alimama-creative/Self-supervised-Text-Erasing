# from __future__ import print_function
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import inspect, re
import numpy as np
import torch
import os
import cv2
import collections
from torch.optim import lr_scheduler
import torch.nn.init as init
from io import BytesIO
import requests
import imutils
import json
import math
import util.text_utils as tu
from util.colorize3_poisson import Colorize

from numpy import random
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def tensor2im_without_norm(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0 + 0.5
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def atten2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor[0]
    image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy/(image_numpy.max()/255.0)
    return image_numpy.astype(imtype)

def latent2im(image_tensor, imtype=np.uint8):
    # image_tensor = (image_tensor - torch.min(image_tensor))/(torch.max(image_tensor)-torch.min(image_tensor))
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def max2im(image_1, image_2, imtype=np.uint8):
    image_1 = image_1[0].cpu().float().numpy()
    image_2 = image_2[0].cpu().float().numpy()
    image_1 = (np.transpose(image_1, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_2 = (np.transpose(image_2, (1, 2, 0))) * 255.0
    output = np.maximum(image_1, image_2)
    output = np.maximum(output, 0)
    output = np.minimum(output, 255)
    return output.astype(imtype)

def variable2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].data.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def masktotensor(mask):
    mask_arr = np.array(mask)
    # print(mask_arr.shape)
    mask_arr[mask_arr<10]=0
    mask_arr[mask_arr>=10]=1
    return torch.as_tensor(mask_arr, dtype=torch.float).permute(2,0,1)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)

    return init_fun

def download_image_from_url(url, img_type='img'):
    try:
        response = requests.get(url, timeout=100)
        if img_type == 'img':
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(BytesIO(response.content)).convert('L')
    except:
        response = requests.get(url, timeout=100)
        if img_type == 'img':
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(BytesIO(response.content)).convert('L')
    return img

         
def get_mask(img, label, mode=0, dilate=0):
    gray_img = np.array(img.convert("L"), dtype=np.float32)
    gray_label = np.array(label.convert("L"), dtype=np.float32)
    diff = np.abs(gray_img - gray_label)
    
    diff_threshold = 20
    diff[diff < diff_threshold] = 1
    diff[diff >= diff_threshold] = 0
    diff = (diff * 255).astype("uint8")

    img_h, img_w = diff.shape
    if mode == 1:
        mask = np.ones((img.size[1], img.size[0]))
        thresh = cv2.threshold(diff, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        bbox_size_threshold = 20
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if w*h < bbox_size_threshold:
                continue
            mask[y:y+h, x:x+w] = 0   
        diff = (mask * 255).astype("uint8")
    elif mode == 2:
        # mask = np.ones((img.size[1], img.size[0]))
        mask = Image.new("L",img.size, 255)
        draw = ImageDraw.Draw(mask)
        thresh = cv2.threshold(diff, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        bbox_size_threshold = 20
        for c in cnts:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # print(box)
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            if w*h < bbox_size_threshold:
                continue
            draw.polygon(box, fill=(0))
        diff  = np.array(mask)
        # diff = (mask * 255).astype("uint8")
    

    diff = cv2.erode(diff, np.ones((dilate, dilate), np.uint8))
    return Image.fromarray(np.repeat(diff[...,np.newaxis],3,2))

def gen_raw_mask(size, info, dilate=0):
    masks = info[:-1].split(";")
    mask = np.ones((size[1], size[0]))
    for obj in masks:
        if len(obj.split(","))<4:
            continue
        x1, x2, y1, y2 = obj.split(",")
        mask[int(y1):int(y2), int(x1):int(x2)] = 0
    diff = (mask * 255).astype("uint8")
    diff = cv2.erode(diff, np.ones((dilate*2, dilate), np.uint8))
    return Image.fromarray(np.repeat(diff[...,np.newaxis],3,2))


Font_Dir = "./data/Items/Font/"
Font_type = ["Alibaba-PuHuiTi-B.ttf", "Alibaba-PuHuiTi-Medium.ttf", "Alibaba-PuHuiTi-Regular.ttf",
            "Alibaba-PuHuiTi-H.ttf", "FZBaiZRZTJW.TTF", "FZFWTongQPOPTJW.TTF", "FZFWZhuZLSHTJWB.TTF", 
            "Alibaba-PuHuiTi-Light.ttf", "FZHuaSJSJW-R.TTF", "FZLiuBSLSJW.TTF", "FZsong.ttf", "JCyuan.ttf"]

def img_mser(img, info, no, num):
    masks = info[:-1].split(";")
    # print(len(masks), no)
    x1, x2, y1, y2 = masks[no].split(",")
    img = img[max(0, int(y1)-2): min(img.shape[0]-1, int(y2)+2), max(0, int(x1)-1): min(img.shape[1]-1, int(x2)+1)]
    word_type = 0
    if int(y2)-int(y1) > int(x2)-int(x1):
        word_type = 1

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    mser = cv2.MSER_create(7, 5, 14400, 0.5)
    regions, boxes = mser.detectRegions(gray)

    areas = np.zeros(len(regions))
    infos = []
    cnt= 0
    for obj in regions:
    # print(obj.shape, obj.reshape(-1, 1, 2).shape)
        hul = cv2.convexHull(obj.reshape(-1, 1, 2))
        x, y, w, h = cv2.boundingRect(hul)
        if word_type == 0 and w > 2*img.shape[1]/num:
            continue
        if word_type == 1 and h > 2*img.shape[0]/num:
            continue
        if w*h > img.shape[0]*img.shape[1]*0.3:
            continue
        areas[cnt]= w*h
        cnt+=1
        infos.append([[x,y,w,h],np.array(obj)])

    areas = areas[:cnt]
    so_ind = np.argsort(areas)
    so_ind = np.flip(so_ind)
    mask = np.ones((img.shape[0], img.shape[1]))
    li = []
    for i in so_ind:
        info = infos[i]
        x,y,w,h = info[0]
        in_f = False
        for l in li:
            if x>l[0]-2 and y>l[1]-2 and x+w<l[0]+l[2]+2 and y+h<l[1]+l[3]+2:
                in_f = True
                break
        if in_f:
            continue
        li.append([x, y, w, h])
        for pos in info[1]:
            # print(pos)
            mask[pos[1],pos[0]] = 0

    return mask, img

def set_font(color, size, image, x, y, direct, config):
    c_n = config["C"]
    if config["C"] >= len(color):
        c_n = len(color)-1
    rgb = color[c_n].split(",")
    rgb = (int(rgb[0]),int(rgb[1]),int(rgb[2]))

    font_size = int(size)

    # defined size
    if config["S"] == 0:
        # font_size = max(9, int(size) - 3 - int(np.random.rand()*3))
        font_size = max(9, int(size) - 3 - int(np.random.rand()*12))
    elif config["S"] == 1:
        font_size = int(size)
    elif config["S"] == 2:
        # font_size = int(size) + 3 + int(np.random.rand()*3)
        font_size = int(size) + 3 + int(np.random.rand()*12)
    elif config["S"] == 3:
        # font_size = int(size) + 8 + int(np.random.rand()*5)
        font_size = int(size) + 8 + int(np.random.rand()*15)
    elif config["S"] == 4:
        # font_size = max(6, int(size) - 8 - int(np.random.rand()*5))
        font_size = max(6, int(size) - 8 - int(np.random.rand()*15))
    # choice of font type
    font = Font_Dir+Font_type[config["F"]]     
    
    c_A1 = 0
    c_A2 = 0
    theta = 0
    shift = 6
    shadow_guassian = 0
    shadow_alpha = 0
    shadow_fill = 0  
    stroke_width = 0
    stroke_fill = 0
    base_size= 20
    # choice of copy or not copy
    copy_type = 0
    if int(size)>base_size and config["T"]==0:
        copy_type = 1
    elif config["A"] == 0:
        c_A1 = 0
        c_A2 = 0
    else:
        # choice of art type
        c_A1 = 0
        c_A2 = 0
        if font_size>base_size and config["A1"]==1: 
            # stroke art type
            c_A1 = 1
            # choice of stroke color
            stroke_c = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
            bg_c = image[y,x]
            if 0.11*rgb[0] + 0.59*rgb[1] + 0.3*rgb[2] < 128:
                if config["A1_C"] == 0:
                    stroke_fill = (255,255,255)
                elif config["A1_C"] == 1:
                    stroke_fill = stroke_c
                elif config["A1_C"] == 2:
                    stroke_fill = (int(0.3*rgb[0]+0.7*bg_c[0]),int(0.3*rgb[1]+0.7*bg_c[1]),int(0.3*rgb[2]+0.7*bg_c[2]))
            else:
                if config["A1_C"] == 0:
                    stroke_fill = (0,0,0) 
                elif config["A1_C"] == 1:
                    stroke_fill = stroke_c
                elif config["A1_C"] == 2:
                    stroke_fill = (int(0.3*rgb[0]+0.7*bg_c[0]),int(0.3*rgb[1]+0.7*bg_c[1]),int(0.3*rgb[2]+0.7*bg_c[2]))
            # choice of stroke width
            stroke_width_list = [2,5,7,9,11,14]
            stroke_width = stroke_width_list[config["A1_S"]]
        if font_size>base_size and config["A2"]==1:
            # shadow art type
            c_A2 = 1
            shadow_c = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
            if config["A2_C"] == 0:
                shadow_fill = (0,0,0) 
            elif config["A2_C"] == 1:
                shadow_fill = shadow_c
            elif config["A2_C"] == 2:
                bg_c = image[y,x]
                shadow_fill = (int(0.35*bg_c[0]),int(0.35*bg_c[1]),int(0.35*bg_c[2]))

            # choice of shadow position
            theta = np.pi/4 * (config["A2_R"] + np.random.rand() - 0.5)
            shift_list = [1,3,5,7,9,12,15,18]
            shift = np.random.randint(shift_list[config["A2_D"]],shift_list[config["A2_D"]+1])
            guassian_list = [0,6,9,12,15]
            alpha_list = [1.0,0.9,0.7,0.5,0.3]
            shadow_guassian = guassian_list[config["A2_G"]]
            shadow_alpha = alpha_list[config["A2_A"]]


    text_info = {
        "mser": copy_type,
        "font": font, 
        "size": font_size,
        "stroke": c_A1, 
        "shadow": c_A2, 
        "rgb": rgb, 
        "alpha": config["Alpha"],
        "warp_perspective": False if config["Warp"]==0 else True,
        "text_gaussian": config["Gau"],
        "shadow_theta": theta, 
        "shadow_shift": shift, 
        "shadow_fill": shadow_fill,
        "shadow_gaussian": shadow_guassian, 
        "shadow_alpha": shadow_alpha, 
        "stroke_width": stroke_width, 
        "stroke_fill": stroke_fill,
        "poisson": True if config["Poi"]==1 else False,
        "underline": False,
        "strong": False,
        "oblique": True if config["Ital"]==1 else False,
        "direction": int(direct),
        "curved": config["Cur"]
    }
    return text_info

def random_gen_config(space_type):
    with open("./util/space_config.json",  "r") as f:
        space = json.load(f)["range"][space_type]
    # print(space)
    config = {}
    # params = ["T","F","C","S","Gau","Poi","Ital","Cur","A","A1","A1_S","A1_C","A2","A2_R","A2_D","A2_C","A2_G","A2_A"]
    for param in space:
        config[param[0]] = np.random.randint(param[1],param[2])

    if space_type == "random8":
        config["Alpha"] = config["Poi"]

    # words = ["T", "Alpha", "Warp", "Poi"]
    # mati = [0.2, 0.3, 0.2, 0.2]
    # # words = ["T"]
    # # mati = [0.2]
    # for i in range(len(words)):
    #     if np.random.rand()<mati[i]:
    #         config[words[i]] = 0
    #     else:
    #         config[words[i]] = 1

    # config["T"] = 0 
    config["A"] = 0
    # config["Alpha"], config["Gau"], config["Poi"] = 1, 1, 1
    # config["Alpha"], config["Gau"], config["Poi"] = 0, 0, 0
    config["Warp"] = 0
    config["Ital"], config["Cur"] = 0, 0
    # config["A1"] = 1
    # config["S"] = 1
    # config["A2"] = 

    return config

def gen_valid_config(index, space_type):
    with open("./util/space_config.json",  "r") as f:
        space = json.load(f)["range"][space_type]
    config = {}
    for param in space:
        config[param[0]] = np.random.randint(param[1],param[2])
    if space_type == "specific1":
        if index%2==1:
            config["T"] = 0
        elif index%8==0:
            config["A1"] = 0
            config["A"] = 1
        elif index%8==2:
            config["A2"] = 0
            config["A"] = 1
        else:
            config["A"] = 0    
    elif space_type == "specific2":
        if index%2==1:
            config["T"] = 0
        elif index%16==0 or index%16 ==8:
            config["A"] = 0
            config["Cur"] = 0
        elif index%16==2 or index%16 ==10:
            config["A"] = 0
            config["Cur"] = 1
        elif index%16==4:
            config["A1"] = 1
            config["A"] = 1
        elif index%16==6:
            config["A2"] = 1
            config["A"] = 1
        elif index%16==12:
            config["A1"] = 1
            config["A"] = 1
            config["Cur"] = 1
        elif index%16==14:
            config["A2"] = 1
            config["A"] = 1
            config["Cur"] = 1
    elif space_type == "specific3":
        config["T"] = 1
        config["A"] = 1
        config["Alpha"] = 1
        config["Gau"] = 1
        config["Poi"] = 1
        if np.random.rand() < 0.2:
            config["Ital"] = 1
        if np.random.rand() < 0.2:
            config["Cur"] = 1
        if np.random.rand() < 0.15:
            config["A1"] = 1
        if np.random.rand() < 0.15:
            config["A2"] = 1

    return config

def gen_config_from_parse(policys, space_type):
    configs = []
    with open("./util/space_config.json",  "r") as f:
        space = json.load(f)["shift"][space_type]
    for policy in policys:
        if len(policy) == 1:
            config = int(policy[0])
        else:
            config = {}
            cnt = 0
            for param in space:
                if param[1] == 0:
                    config[param[0]] = int(policy[cnt])
                    cnt += 1
                elif param[1] == -1:
                    config[param[0]] = int(param[2])
                elif param[1] == 1:
                    config[param[0]] = int(param[2]) + int(policy[cnt])
                    cnt += 1
        configs.append(config)
    return configs


def feather(text_mask, level):
    # determine the gaussian-blur std:
    if level == 0:
        return text_mask
    elif level == 1:
        bsz = 0.25
        ksz=1
    elif level == 2:
        bsz = max(0.30, 0.5 + 0.1*np.random.randn())
        ksz = 3
    else:
        bsz = max(0.5, 1.5 + 0.5*np.random.randn())
        ksz = 5
    return cv2.GaussianBlur(text_mask,(ksz,ksz),bsz)

def draw_text(image, pos, word, text_info):
    text_renderer = tu.RenderFont()
    colorizer = Colorize()
    font = text_renderer.init_font(text_info)
    mask = np.zeros((image.shape[0],image.shape[1]))
    try:
        img, bb = text_renderer.render_sample(word, font, mask, pos, text_info)
        img= feather(img, text_info["text_gaussian"])
        im_final = colorizer.color(image,[img],text_info)
    except:
        im_final = image
    return im_final
    
def draw_text_with_pillow(image, pos, sentence, text_info):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(text_info["font"], text_info["size"])
    x, y = pos
    [shadow_x, shadow_y] = text_info["shadow_shift"] * np.array([-np.sin(text_info["shadow_theta"]), np.cos(text_info["shadow_theta"])])
    shadow_fill = text_info["shadow_fill"]
    stroke_width = text_info["stroke_width"]
    stroke_fill = text_info["stroke_fill"]
    rgb = text_info["rgb"]
    c_A1, c_A2 = text_info["stroke"], text_info["shadow"]
    if int(text_info["direction"])==1.0:
        _, fh = font.getsize(sentence[0])
        for k,s in enumerate(sentence):
            if c_A1 == 1 and c_A2 == 1:
                draw.text((x-shadow_x, y+fh*k-shadow_y), s, fill=shadow_fill, font=font, stroke_width=stroke_width, stroke_fill=shadow_fill)
                draw.text((x, y+fh*k), s, rgb, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
            elif c_A1 == 1:
                # print("art_type 1")
                draw.text((x, y+fh*k), s, rgb, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
            elif c_A2 == 1:
                draw.text((x-shadow_x, y+fh*k-shadow_y), s, fill=shadow_fill, font=font)
                draw.text((x, y+fh*k), s, rgb, font=font)
            else:
                draw.text((x, y+fh*k), s, rgb, font=font)
    else: 
        if c_A1 == 1 and c_A2 == 1:
            draw.text((x-shadow_x, y-shadow_y), sentence, fill=shadow_fill, font=font, stroke_width=stroke_width, stroke_fill=shadow_fill)
            draw.text((x, y), sentence, rgb, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
        elif c_A1 == 1:
            # print("art_type 1")
            draw.text((x, y), sentence, rgb, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
        elif c_A2 == 1:
            draw.text((x-shadow_x, y-shadow_y), sentence, fill=shadow_fill, font=font)
            draw.text((x, y), sentence, rgb, font=font)
        else:
            draw.text((x, y), sentence, rgb, font=font)      
    image = np.array(image)   
    return image

def generate_img(image, info, config, random_index=True, verbose=False):
    image = np.array(image)
    mask_info = info["mask"]
    info = info["place"]
    # draw = ImageDraw.Draw(image)
    if len(info['text']) == 0:
        return Image.fromarray(image)
    # choice of position
    if random_index:
        index = np.random.randint(0,len(info['text']))
    else:
        index = 0
    anchor_color = info['text'][index]

    obj = info['obj'][:-1].split(";")
    for j, p in enumerate(anchor_color):
        p = obj[j]
        if len(p.split(",")) <3:
            continue
        s = p.split(",")
        if len(s) == 3:
            sentence, size, direction = s
        else:
            sentence = ""
            for k in range(len(s)-3):
                sentence += s[k]+","
            sentence+=s[-3]
            size=s[-2]
            direction=s[-1]
        if len(sentence) == 0:
            print("blank sentences")
            continue
        x, y, color = anchor_color[j][0], anchor_color[j][1], anchor_color[j][2]
        text_info = set_font(color, size, image, x, y, direction, config)
        if text_info["mser"] == 1:
            mask, vis = img_mser(image, mask_info, j, len(sentence)) 
            begin_x, begin_y = max(x-5,0), max(0,y-10)
            for i in range(begin_x, min(begin_x+vis.shape[1], image.shape[1])):
                for m in range(begin_y, min(begin_y+vis.shape[0], image.shape[0])):
                    if mask[m-begin_y,i-begin_x]==0:
                       image[m,i] = [vis[m-begin_y,i-begin_x][2], vis[m-begin_y,i-begin_x][1], vis[m-begin_y,i-begin_x][0]]
        else:
            image = draw_text(image, (x,y), sentence, text_info) 
            # image = draw_text_with_pillow(image, (x,y), sentence, text_info) 
    if verbose:
        print(text_info)
    # im_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im_rgb = Image.fromarray(image)
    return im_rgb

def generate_valid_img(image, info, index, gen_space):
    config = gen_valid_config(index, gen_space)
    try:
        return generate_img(image, info, config, False)
    except:
        return image

def generate_img_with_config(image, info, config, random_index=True, verbose=False):
    try:
        return generate_img(image, info, config, random_index, verbose)
    except:
        return image

import math
def mse_psnr(img1, img2):
    mse = ((img1 - img2)**2).mean()
    if mse == 0:
        return mse, 100
    # print(mse)
    psnr = 10 * math.log10(1/mse)
    return mse, psnr