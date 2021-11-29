import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image 
import torchmetrics
from sklearn.metrics import jaccard_score as metric

def normalize(x):
        x_ = torch.flatten(x, -2, -1) # B, C, H, W -> B, C, HW
        
        x_max = x_.max(-1)[0].view(x.size(0), 3, 1, 1)
        x_min = x_.min(-1)[0].view(x.size(0), 3, 1, 1)
      
        x_out = (x - x_min) / (x_max - x_min)
      
        return x_out  # B, C, H, W
    
def clip(img, img_min=None, img_max=None):
    if img_min is None:
        img_min = torch.tensor([-2.1179, -2.0357, -1.8044]).view(1,3,1,1).cuda()

    if img_max is None:
        img_max = torch.tensor([2.2489, 2.4286, 2.6400]).view(3,1,1).cuda()

    img = torch.clip(img, min=img_min, max=img_max)
    
    return img
    
def process_grad(grad):
    grad = normalize(grad.abs())
    spatial = grad.mean(dim=1, keepdim=True) * torch.randn_like(grad)  # B, C, H, W
    
    channel = F.adaptive_avg_pool2d(grad, 1)  # B, C, 1, 1
    channel = channel * torch.randn_like(channel) * 5.0  # B, C, 1, 1
    
    return spatial + channel 
    
    
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def iou(output, target, num_classes, ignore_index=255):
    mIoU = metric(target.view(-1).cpu(), output.view(-1).cpu(), average='macro')
    
    return mIoU
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def transform_color(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 15,
        14: 17,
        15: 18,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()
    
    
def get_color_pallete(npimg, dataset='city'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'city':
        cityspallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
        #         palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250,
        #         170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
        #         0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        #         zero_pad = 256 * 3 - len(palette)
        #         for i in range(zero_pad):
        #             palette.append(0)
        out_img.putpalette(cityspallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img
    