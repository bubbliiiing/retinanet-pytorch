#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
import torch

from nets.retinanet import Resnet, Retinanet

if __name__ == '__main__':
    inputs = torch.randn(5, 3, 512, 512)
    # Test inference
    model = Retinanet(80,2)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    
