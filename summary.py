#--------------------------------------------#
#   该部分代码用于看网络参数
#--------------------------------------------#
from nets.retinanet import retinanet

if __name__ == '__main__':
    model   = retinanet(80, 2)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    
