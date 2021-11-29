import argparse
import os
import datetime
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.distributed
import torch.utils.data
import torch.backends.cudnn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101

from custom import FCN_ResNet50
from datasets.synthia import *
from datasets.gtav import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=40,
                    help='Number of traning epochs')
parser.add_argument('--baselr', type=float, default=2e-4,
                    help='Base learning rate.')
parser.add_argument('--eps', type=float, default=0.5,
                    help='Perturbation Strength.')
parser.add_argument('--numclasses', type=int, default=16,
                    help='Number of classes. Default=16')
parser.add_argument('--batch', type=int, default=16,
                    help='Batch Size. Default=16')
parser.add_argument('--cuda_group', type=str, default="0, 1")

parser.add_argument('--wide', action='store_true', default=False,
                    help='Use wider ResNet')                   
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Pretrained ResNet')
parser.add_argument('--robust', action='store_true', default=False,
                    help='Our method')
parser.add_argument('--resnet50', action='store_true', default=False,
                    help='ResNet-50 as backbone')
parser.add_argument('--resnet101', action='store_true', default=False,
                    help='ResNet-101 as backbone')
parser.add_argument('--vgg', action='store_true', default=False,
                    help='VGG-16 as backbone')
parser.add_argument('--augment', action='store_true', default=False,
                    help='Data augmentation')
                    
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--dataroot', default='/home/wei/data2/Dataset/SemanticSeg/SYNTHIA/',
                    help='Root directory to traning data.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_group
torch.manual_seed(0)
torch.cuda.manual_seed(0)

print(">>> [INFO] Robust Training: {}".format(args.robust))
print(">>> [INFO] Epsilon: {}".format(args.eps))
print(">>> [INFO] Pretrained ResNet: {}".format(args.pretrained))
print(">>> [INFO] Learning Rate: {}".format(args.baselr))
print(">>> [INFO] Data Augmentation: {}".format(args.augment))
print(">>> [INFO] Saving checkpoints to: {}".format(args.savemodel))

# Setup dataloader 
#print(">>> [INFO] Loading SYNTHIA data...")
#[trainList, testList] = synthiaList(args.dataroot)

#trainLoader = Data.DataLoader(synthiaDataSet(data_root=args.dataroot, data_list=trainList, max_iters=None, train=True, num_classes=args.numclasses, augment=args.augment),
#                              batch_size=args.batch, shuffle=True, num_workers=8, drop_last=False)
                              
#testLoader = Data.DataLoader(synthiaDataSet(data_root=args.dataroot, data_list=testList, train=False, num_classes=args.numclasses),
#                              batch_size=args.batch, shuffle=False, num_workers=8, drop_last=False)
                              
print(">>> [INFO] Loading GTAV data...")
dataroot = '/home/wei/data2/Dataset/SemanticSeg/GTA/'
[trainList, testList] = gtaList(dataroot)

trainLoader = Data.DataLoader(GTAVDataSet(data_root=dataroot, data_list=trainList, max_iters=None, train=True, num_classes=args.numclasses, augment=args.augment),
                              batch_size=args.batch, shuffle=True, num_workers=8, drop_last=False)
                              
testLoader = Data.DataLoader(GTAVDataSet(data_root=dataroot, data_list=testList, train=False, num_classes=args.numclasses),
                              batch_size=args.batch, shuffle=False, num_workers=8, drop_last=False)

# Setup model and optimizer 
print(">>> [INFO] Setting up models...")
if args.resnet50:
    print(">>> [INFO] ResNet-50 as BackBone")
    url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    #model = fcn_resnet50(pretrained=False, num_classes=args.numclasses)
    model = FCN_ResNet50(pretrained=True, num_classes=args.numclasses, wide=args.wide, robust=args.robust)
elif args.resnet101:
    print(">>> [INFO] ResNet-101 as BackBone")
    url = 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    model = fcn_resnet101(pretrained=False, num_classes=args.numclasses)
else:
    raise Exception("Invalid model selected. Choose between resnet101/resnet50.")
    
    
if args.pretrained:
    # Load ImageNet-pretrained ResNet weights to feature extractor 
    ckpt = torch.hub.load_state_dict_from_url(url)
    # Remove unmatched FC keys 
    ckpt_dup = ckpt.copy()
    
    for key in ckpt.keys():
        if key.split('.')[0] == 'fc':
            ckpt_dup.pop(key)
    
    model.backbone.load_state_dict(ckpt_dup)
    print('>>> [INFO] Loaded ImageNet Pretrained Weights')
    
model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None:
    ckpt = torch.load(args.loadmodel)
    model.load_state_dict(ckpt['state_dict'])
    print(">>> [INFO] Loaded Pre-trained weights from {}".format(args.loadmodel))

if args.pretrained:
  for module in model.module.backbone.modules():
      # print(module)
      if isinstance(module, nn.BatchNorm2d):
          if hasattr(module, 'weight'):
              module.weight.requires_grad_(False)
          if hasattr(module, 'bias'):
              module.bias.requires_grad_(False)
          module.eval()

optim = optim.Adam(model.parameters(), lr=args.baselr, betas=(0.9,0.999))

criterion = nn.CrossEntropyLoss(ignore_index=255)

def adjust_learning_rate(iters, max_iters=32000, base_lr=0.01, power=0.9):
    lr = base_lr * ((1 - float(iters) / max_iters) ** power)
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def consistency_loss(feats1, feats2):
    loss = []
    
    for ii, (feat1, feat2) in enumerate(zip(feats1, feats2)):
        loss.append(torch.mean((feat1 - feat2).pow(2)))
    
    loss = sum(loss)
        
    return loss
    
def grad_norm(grad, eps=0.5):
        grad = grad.pow(2)
        grad = F.normalize(grad, p=2, dim=1) 
        grad = grad * eps
          
        return grad 
        
def clip(img, img_min=None, img_max=None):
        if img_min is None:
            img_min = torch.tensor([-2.1179, -2.0357, -1.8044]).view(1,3,1,1).cuda()

        if img_max is None:
            img_max = torch.tensor([2.2489, 2.4286, 2.6400]).view(3,1,1).cuda()

        img = torch.clip(img, min=img_min, max=img_max)
        
        return img
    
# Train and test functions 
def train(imgs, labels):
    model.train()
    loss_const = torch.zeros(1).cuda()
    
    if args.robust:
        '''
        # Compute gradient 
        #imgs_temp = imgs.clone().detach()
        #imgs_temp.requires_grad = True
        
        #model.eval()
        #feat_temp = model.module.backbone(imgs_temp)['out']
        #grads = torch.autograd.grad(outputs=feat_temp, inputs=imgs_temp, grad_outputs=torch.ones_like(feat_temp), create_graph=False)
        #grads = grad_norm(grads[0], eps=args.eps)
        #model.zero_grad()
        
        #imgs_scp = imgs.clone()
        #imgs_scp = imgs_scp + grads
        #imgs_scp = clip(imgs_scp).detach()
        
        #del imgs_temp, feat_temp, grads
        
        # Forward-pass
        model.train()
        feats_clean = model.module.backbone(imgs)['out']
        feats_scp = model.module.backbone(imgs_scp)['out']
        
        #preds_clean = model(imgs)['out']
        #preds_scp = model(imgs_scp)['out']
        #loss_CE = criterion(preds_scp, labels) + criterion(preds_clean, labels)
        
        preds = model(imgs_scp)['out']
        loss_CE = criterion(preds, labels)
        
        loss_const = consistency_loss(feats_scp, feats_clean)
        loss = loss_CE + 0.1*loss_const'''
        model.train()
        preds, [feats_clean, feats_scp] = model(imgs)
        
        loss_CE = criterion(preds, labels)
        loss_const = consistency_loss(feats_scp, feats_clean)
        loss = loss_CE + 0.1*loss_const  
        
    else:
        preds = model(imgs)#['out']
        loss = criterion(preds, labels) 
    
    optim.zero_grad()

    loss.backward()
    
    optim.step()
    
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    output = preds.max(1)[1]
    intersection, union, target = intersectionAndUnionGPU(output, labels, args.numclasses, 255)
    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    return loss.item(), loss_const.item(), mIoU, mAcc, allAcc
    
def test(imgs, labels):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    model.eval()
    with torch.no_grad():
        preds = model(imgs)#['out']
    loss = criterion(preds, labels) 
    
    # M-IOU
    output = preds.max(1)[1]
    intersection, union, target = intersectionAndUnionGPU(output, labels, args.numclasses, 255)
    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    return loss.item(), mIoU, mAcc, allAcc

# main function 
def main():
    EPOCHS = args.epoch
    train_loss = []
    test_loss = []
    start_full_time = time.time() 
    
    for epoch in range(1, EPOCHS+1):
        total_train_loss = 0
        total_test_loss = 0
        total_mIoU = 0
        total_mAcc = 0
        total_allAcc = 0
        start_time = time.time()
        
        print(">>> [INFO] Begin Training...")
        # TRAINING
        for batch_idx, all_data in enumerate(trainLoader):
            adjust_learning_rate(iters=batch_idx, max_iters=30000, base_lr=args.baselr)
            images, labels, names = all_data
            images, labels = images.to('cuda'), labels.long().to('cuda')
            tic = time.time()
            model.train()
            loss, loss_reg, mIoU, mAcc, allAcc = train(images, labels)
            toc = time.time()
            
            total_train_loss += loss 
       
            print('[INFO] Epoch: %d/%d  Iter: %d/%d  CE Loss: %.3f  Reg Loss: %.3f  mIoU: %.3f  mAcc: %.3f  allAcc: %.3f  Time: %.2f'
                  % (epoch, EPOCHS, batch_idx+1, len(trainLoader), loss, loss_reg, mIoU, mAcc, allAcc, toc-tic))
            
        print('[INFO] Epoch: %d/%d   Total Training Loss: %.3f  Time/Epoch: %.2f hrs' %
              (epoch, EPOCHS, total_train_loss / len(trainLoader), (time.time()-start_time)/3600))
        #train_loss.append(total_train_loss / len(trainLoader))
        
        
        print(">>> [INFO] Begin Testing...")
        # TESTING
        for batch_idx, all_data in enumerate(testLoader):
            images, labels, names = all_data
            images, labels = images.to('cuda'), labels.long().to('cuda')
            tic = time.time()
            model.eval()
            test_loss, mIoU, mAcc, allAcc = test(images, labels)
            toc = time.time()
            
            total_test_loss += test_loss 
            total_mIoU += mIoU
            total_mAcc += mAcc
            total_allAcc += allAcc
            
            print('[INFO] Epoch: %d/%d  Iter: %d/%d  CE Loss: %.3f  mIoU: %.3f  mAcc: %.3f  allAcc: %.3f  Time: %.2f'
                  % (epoch, EPOCHS, batch_idx+1, len(testLoader), test_loss, mIoU, mAcc, allAcc, toc-tic))
            
        print('[INFO] Epoch: %d/%d   Total Testing Loss: %.3f  Total mIoU: %.3f  Total mAcc: %.3f  Total allAcc: %.3f  Time/Epoch: %.2f hrs' %
              (epoch, EPOCHS, total_test_loss / len(testLoader), total_mIoU / len(testLoader), total_mAcc / len(testLoader), total_allAcc / len(testLoader), (time.time()-start_time)/3600))
        #test_loss.append(total_test_loss / len(testLoader))
        
        if args.robust:
            #SAVE
            if args.vgg:
              savefilename = os.path.join(args.savemodel, "vgg_robust_" + str(epoch) + ".tar")
            elif args.resnet50:
              savefilename = os.path.join(args.savemodel, "resnet50_robust_" + str(epoch) + ".tar")
            else:
              savefilename = os.path.join(args.savemodel, "resnet101_robust_" + str(epoch) + ".tar")
              
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(trainLoader),
                'test_loss': total_test_loss / len(testLoader),
                'mIoU': total_mIoU / len(testLoader),
                'mAcc': total_mAcc / len(testLoader),
                'allAcc': total_allAcc / len(testLoader)
            }, savefilename)
        
        else:
            #SAVE
            if args.vgg:
              savefilename = os.path.join(args.savemodel, "vgg_baseline_" + str(epoch) + ".tar")
            elif args.resnet50:
              savefilename = os.path.join(args.savemodel, "resnet50_baseline_" + str(epoch) + ".tar")
            else:
              savefilename = os.path.join(args.savemodel, "resnet101_baseline_" + str(epoch) + ".tar")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(trainLoader),
                'test_loss': total_test_loss / len(testLoader),
                'mIoU': total_mIoU / len(testLoader),
                'mAcc': total_mAcc / len(testLoader),
                'allAcc': total_allAcc / len(testLoader)
            }, savefilename)
            
            
    print('>>> [INFO] Training Completed. Total Time = %.2f HRs' % ((time.time() - start_full_time) / 3600))
    
if __name__ == '__main__':
    main()

