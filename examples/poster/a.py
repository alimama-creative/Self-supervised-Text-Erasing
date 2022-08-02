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

i = 0
with open("ps_valid.txt", "r") as f:
    for l in f.readlines():
        seq = l.split("\t")
        im = download_image_from_url(seq[1])
        im.save("valid/%d.png"%i)

        im = download_image_from_url(seq[3].strip("\n"))
        
        im.save("valid/%d_g.png"%i)

        i+=1