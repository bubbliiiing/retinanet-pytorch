import numpy as np
from PIL import Image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def preprocess_input(image):
    image   /= 255
    mean    = (0.406, 0.456, 0.485)
    std     = (0.225, 0.224, 0.229)
    image   -= mean
    image   /= std
    return image

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        '0': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
        '1': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
        '2': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        '3': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
        '4': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)