import requests
from PIL import Image
from io import BytesIO

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

i = 1
# s = "https://alimama-creative-public.oss-cn-zhangjiakou.aliyuncs.com/gangwei.jgw/SCUT_Syn/EnsText/train/all_masks/"
with open("train.txt", "r") as f:
    for l in f.readlines():
        # seq = l.split("\t")
        seq = l.split("\t")
        im = download_image_from_url(seq[1].replace("images","masks"))
        im.save("train/all_masks/%d.jpg"%i)

        im = download_image_from_url(seq[1])
        im.save("train/all_images/%d.jpg"%i)
        i+=1
        # im = download_image_from_url(s.replace("masks", "labels")+l.strip())
        # im.save("valid/all_labels/"+l.strip())
        # im = download_image_from_url(seq[3].strip("\n"))
        
        # im.save("valid/%d_g.png"%i)

        # i+=1