# TEST MODEL ON CITYSCAPE DATASET
import argparse
import os
import datetime
import logging
import time
import pandas as pd 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.distributed
import torch.utils.data
import torch.backends.cudnn
import torch.nn.functional as F
import torchvision.models as models
from custom import FCN_ResNet50, FCN_ResNet101, VGG16
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101

from datasets.cityscapes import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default='0', type=str, help='Select GPU')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--resnet50', action='store_true', default=False,
                    help='ResNet-50 as backbone')
parser.add_argument('--resnet101', action='store_true', default=False,
                    help='ResNet-101 as backbone')
parser.add_argument('--wide', action='store_true', default=False,
                    help='Use wider ResNet') 
parser.add_argument('--vgg', action='store_true', default=False,
                    help='VGG-16 as backbone')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Setup dataloader 
print(">>> [INFO] Loading data...")
data_root = '/home/wei/data2/Dataset/SemanticSeg/cityscape/'
testList = cityscapesList(data_root)                              
testLoader = Data.DataLoader(cityscapesDataSet(data_root=data_root, data_list=testList, num_classes=16),
                              batch_size=1, shuffle=False, num_workers=8, drop_last=False)

# Setup model and optimizer 
print(">>> [INFO] Setting up models...")
if args.resnet50:
    print(">>> [INFO] ResNet-50 as BackBone")
    model = FCN_ResNet50(pretrained=False, num_classes=19, wide=args.wide)
    #model = fcn_resnet50(pretrained=False, num_classes=19)
elif args.resnet101:
    print(">>> [INFO] ResNet-101 as BackBone")
    model = FCN_ResNet101(pretrained=False, num_classes=16, wide=args.wide)
elif args.vgg:
    model = VGG16(pretrained=True, num_classes=16)
else:
    raise Exception("Invalid model selected. Choose between resnet101/resnet50.")
model = nn.DataParallel(model)
model.cuda()

# Load pretrained weights
print(">>> [INFO] Loading pre-trained weights to models...")
ckpt = torch.load(args.loadmodel)
model.load_state_dict(ckpt['state_dict'])

def test(imgs, labels, name):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    model.eval()
    with torch.no_grad():
        preds = model(imgs)#['out']
        
    pixAcc = (preds.max(1)[1] == labels).sum()/(preds.size(0)*preds.size(2)*preds.size(3))
    
    # M-IOU
    output = preds.max(1)[1]
    intersection, union, target = intersectionAndUnionGPU(output, labels, 16, 255)
    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    '''# save the result 
    preds = preds.cpu().numpy().squeeze().argmax(0)
    preds = transform_color(preds)
    mask = get_color_pallete(preds, "city")
    mask_filename = name[0] if len(name[0].split("/")) < 2 else name[0].split("/")[1]
    if mask.mode == 'P':
        mask = mask.convert('RGB')
    mask.save(os.path.join("./results/", mask_filename))'''
    
    #mIoU = iou(output.view(-1).long(), labels.view(-1), num_classes=16, ignore_index=255)
    
    return mIoU, mAcc, allAcc, pixAcc.item()
    
    
print(">>> [INFO] Begin Testing...")
# TESTING
total_mIoU = 0
total_mAcc = 0
total_allAcc = 0
total_pixAcc = 0
start_time = time.time()

collect_names = []
collect_mIoU = []
collect_pixAcc = []

for batch_idx, all_data in enumerate(testLoader):
    images, labels, names = all_data
    images, labels = images.to('cuda'), labels.long().to('cuda')
    tic = time.time()
    model.eval()
    mIoU, mAcc, allAcc, pixAcc = test(images, labels, names)
    toc = time.time()
    
    collect_names.append(names)
    collect_mIoU.append(mIoU)
    collect_pixAcc.append(pixAcc)
    
    total_mIoU += mIoU
    total_mAcc += mAcc
    total_allAcc += allAcc
    total_pixAcc += pixAcc
    
    print('[INFO]Iter: %d/%d mIoU: %.3f  mAcc: %.3f  allAcc: %.3f  pixAcc: %.3f  Time: %.2f'
          % (batch_idx+1, len(testLoader), mIoU, mAcc, allAcc, pixAcc, toc-tic))
    
print('[INFO] Total mIoU: %.3f  Total mAcc: %.3f  Total allAcc: %.3f  Total pixAcc: %.3f  Time/Epoch: %.2f hrs' %
      (total_mIoU / len(testLoader), total_mAcc / len(testLoader), total_allAcc / len(testLoader), total_pixAcc/len(testLoader), (time.time()-start_time)/3600))
      
df = pd.DataFrame({"Name": collect_names, 
                   "mIoU": collect_mIoU,
                   "pixAcc": collect_pixAcc})
                   
df.to_csv("results.csv", index=True, float_format="%.5f")


