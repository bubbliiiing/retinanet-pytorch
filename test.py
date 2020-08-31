import torch
from nets.retinanet import Retinanet
from nets.retinanet import Resnet

if __name__ == '__main__':

    inputs = torch.randn(5, 3, 512, 512)


    # Test inference
    model = Retinanet(80,2)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    