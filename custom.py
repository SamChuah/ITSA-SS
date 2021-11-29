import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import torchvision.models as models 
from torchvision.models.resnet import BasicBlock, ResNet
from utils import normalize, clip, process_grad

class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_classes=16):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(m.in_channels,
                                                            m.out_channels,
                                                            m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
                
    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()
    
    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8
        
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        fc7_response = self.drop7(h)

        h = self.score_fr(fc7_response)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4*0.01)  
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3*0.0001)  
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        return h
        #return x1, x2, x3, x4, x5



class ResNet50(nn.Module):
    def __init__(self, pretrained=True, URL=None):
        super().__init__()
        self.resnet = models.resnet.resnext50_32x4d()
        #self.resnet = models.resnet.resnet50()
        if pretrained:
            if URL is None:
                URL = 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'
                #URL = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
                self.load_ckpt(URL)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x1 = x
        x = self.resnet.bn1(x1)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x2 = x
        x = self.resnet.layer1(x2)
        x3 = self.resnet.layer2(x)
        x4 = self.resnet.layer3(x3)
        x5 = self.resnet.layer4(x4)
        return x1, x2, x3, x4, x5
    
    def load_ckpt(self, ckpt_url):    
        ckpt = torch.hub.load_state_dict_from_url(ckpt_url)   
        self.resnet.load_state_dict(ckpt)
        print('>>> [INFO] Loaded ImageNet pretrained weights')
    
  
class ResNet101(nn.Module):
    def __init__(self, pretrained=True, URL=None):
        super().__init__()
        self.resnet = models.resnet.resnext101_32x8d()
        if pretrained:
            if URL is None:
                URL = 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'
                self.load_ckpt(URL)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x1 = x
        x = self.resnet.bn1(x1)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x2 = x
        x = self.resnet.layer1(x2)
        x3 = self.resnet.layer2(x)
        x4 = self.resnet.layer3(x3)
        x5 = self.resnet.layer4(x4)
        return x1, x2, x3, x4, x5
        
    def load_ckpt(self, ckpt_url):       
        ckpt = torch.hub.load_state_dict_from_url(ckpt_url)   
        self.resnet.load_state_dict(ckpt)
        print('>>> [INFO] Loaded ImageNet pretrained weights')
        

class FCN8_wide(nn.Module):
    def __init__(self, num_classes=16):
        super(FCN8_wide, self).__init__()
        self.fc1 = nn.Conv2d(2048, 4096, 7, 1, 3)
        self.drop1 = nn.Dropout2d()
        
        self.fc2 = nn.Conv2d(4096, 4096, 1, 1, 0, 1)
        self.drop2 = nn.Dropout2d()
        
        self.score_32 = nn.Conv2d(4096, num_classes, 1, 1, 0, 1)
        self.score_16 = nn.Conv2d(1024, num_classes, 1, 1, 0, 1)
        self.score_8 = nn.Conv2d(512, num_classes, 1, 1, 0, 1)
        
        self.upconv_32 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upconv_16 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upconv = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, 0, bias=False)
        
    def forward(self, feats):
        x2, x4, x8, x16, x32 = feats
        
        x = self.fc1(x32)  # [2048, 16, 32]
        x = F.relu(x, inplace=True)
        x = self.drop1(x)  # [4096, 16, 32]
        
        x = self.fc2(x)    # [4096, 16, 32]
        x = F.relu(x, inplace=True)
        x = self.drop2(x)  # [4096, 16, 32]
        
        x = self.score_32(x)   # [10, 16, 32]
        x = self.upconv_32(x)  # [10, 32, 64]
        x_32_up = x
        
        x = self.score_16(x16)  # [10, 32, 64]
        x = x + x_32_up         # [10, 32, 64]
        x_16_up = self.upconv_16(x)  # [10, 64, 128]
        
        x = self.score_8(x8)    # [10, 64, 128]
        x = x + x_16_up
        x = self.upconv(x)
        
        return x  
        
        
class FCN8(nn.Module):
    def __init__(self, num_classes=16):
        super(FCN8, self).__init__()
        self.fc1 = nn.Conv2d(2048, 4096, 3, 1, 1, 1)
        self.drop1 = nn.Dropout2d()
        
        self.fc2 = nn.Conv2d(4096, 4096, 1, 1, 0, 1)
        self.drop2 = nn.Dropout2d()
        
        self.score_32 = nn.Conv2d(4096, num_classes, 1, 1, 0, 1)
        self.score_16 = nn.Conv2d(1024, num_classes, 1, 1, 0, 1)
        self.score_8 = nn.Conv2d(512, num_classes, 1, 1, 0, 1)
        
        self.upconv_32 = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, 1, 1, bias=False)
        self.upconv_16 = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, 1, 1, bias=False)
        self.upconv = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, 0, bias=False)
        
    def forward(self, feats):
        x2, x4, x8, x16, x32 = feats
        
        x = self.fc1(x32)  # [2048, 16, 32]
        x = F.relu(x, inplace=True)
        x = self.drop1(x)  # [4096, 16, 32]
        
        x = self.fc2(x)    # [4096, 16, 32]
        x = F.relu(x, inplace=True)
        x = self.drop2(x)  # [4096, 16, 32]
        
        x = self.score_32(x)   # [10, 16, 32]
        x = self.upconv_32(x)  # [10, 32, 64]
        x_32_up = x
        
        x = self.score_16(x16)  # [10, 32, 64]
        x = x + x_32_up         # [10, 32, 64]
        x_16_up = self.upconv_16(x)  # [10, 64, 128]
        
        x = self.score_8(x8)    # [10, 64, 128]
        x = x + x_16_up
        x = self.upconv(x)
        
        return x  
        
        
class FCN_ResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=16, robust=False, wide=False):
        super().__init__()
        self.robust = robust
        
        # Encoder
        self.encoder = ResNet50(pretrained=pretrained)

        # Decoder 
        #self.decoder = FCN8(num_classes=num_classes)
        if wide:
          self.decoder = FCN8_wide(num_classes=num_classes)
        else:
          self.decoder = FCN8(num_classes=num_classes)
          
    def grad_norm(self, grad):
        grad = grad.pow(2)
        grad = F.normalize(grad, p=2, dim=1) 
        grad = grad * 0.5
          
        return grad 
        
    def clip(self, img, img_min=None, img_max=None):
        if img_min is None:
            img_min = torch.tensor([-2.1179, -2.0357, -1.8044]).view(1,3,1,1).cuda()

        if img_max is None:
            img_max = torch.tensor([2.2489, 2.4286, 2.6400]).view(3,1,1).cuda()

        img = torch.clip(img, min=img_min, max=img_max)
        
        return img
    
    def forward(self, x):
        if self.training:
            if self.robust:
                x_ = x.clone().detach()
                x_.requires_grad = True
                
                self.encoder.eval()
                
                feats = self.encoder(x_)
                temp_feat = []
                
                temp_feat.append(F.interpolate(feats[-1], scale_factor=4, mode='bilinear', align_corners=True))
                temp_feat.append(F.interpolate(feats[-2], scale_factor=2, mode='bilinear', align_corners=True))
                temp_feat.append(feats[-3])
                
                temp_feat = torch.cat(temp_feat, dim=1)
                grad = torch.autograd.grad(outputs=temp_feat, inputs=x_, grad_outputs=torch.ones_like(temp_feat), create_graph=False)
                grad = grad[0].clone().detach() 
                
                grad = self.grad_norm(grad)
                x_scp = x.clone() + grad
                x_scp = self.clip(x_scp).detach()
                
                del grad, temp_feat
                
                self.encoder.train()
                
                feats = self.encoder(x)
                feats_scp = self.encoder(x_scp)
                #out = self.decoder(feats)
                out_scp = self.decoder(feats_scp)
                
                return out_scp, [feats[3:], feats_scp[3:]]
            else:
            
                feats = self.encoder(x)
                out = self.decoder(feats)
                return out
                
        else:
            feats = self.encoder(x)
            out = self.decoder(feats)
            return out

class FCN_ResNet101(nn.Module):
    def __init__(self, pretrained=True, num_classes=16, robust=False):
        super().__init__()
        self.robust = robust
        # Encoder
        self.encoder = ResNet101(pretrained=pretrained)

        # Decoder 
        self.decoder = FCN8(num_classes=num_classes)
        
    def grad_norm(self, grad):
        grad = grad.pow(2)
        grad = F.normalize(grad, p=2, dim=1) 
        grad = grad * 0.2
          
        return grad 
        
    def clip(self, img, img_min=None, img_max=None):
        if img_min is None:
            img_min = torch.tensor([-2.1179, -2.0357, -1.8044]).view(1,3,1,1).cuda()

        if img_max is None:
            img_max = torch.tensor([2.2489, 2.4286, 2.6400]).view(3,1,1).cuda()

        img = torch.clip(img, min=img_min, max=img_max)
        
        return img
    
    def forward(self, x):
        if self.training:
            if self.robust:
                x_ = x.clone().detach()
                x_.requires_grad = True
                
                self.encoder.eval()
                
                feats = self.encoder(x_)
                temp_feat = []
                
                temp_feat.append(F.interpolate(feats[-3], scale_factor=1/4, mode='bilinear', align_corners=True))
                temp_feat.append(F.interpolate(feats[-2], scale_factor=1/2, mode='bilinear', align_corners=True))
                temp_feat.append(feats[-1])
                
                #temp_feat.append(F.interpolate(feats[-1], scale_factor=16, mode='bilinear', align_corners=True))
                #temp_feat.append(F.interpolate(feats[-2], scale_factor=8, mode='bilinear', align_corners=True))
                #temp_feat.append(F.interpolate(feats[-3], scale_factor=4, mode='bilinear', align_corners=True))
                #temp_feat.append(F.interpolate(feats[-4], scale_factor=2, mode='bilinear', align_corners=True))
                #temp_feat.append(feats[0])
                
                temp_feat = torch.cat(temp_feat, dim=1)
                grad = torch.autograd.grad(outputs=temp_feat, inputs=x_, grad_outputs=torch.ones_like(temp_feat), create_graph=False)
                grad = grad[0].clone().detach() 
                
                grad = self.grad_norm(grad)
                x_ = x.clone().detach() + grad
                x_ = self.clip(x_)
                
                self.encoder.train()
                
                feats = self.encoder(x)
                feats_ = self.encoder(x_)
                out = self.decoder(feats_)
                
                return out, [feats, feats_]
            else:
            
                feats = self.encoder(x)
                out = self.decoder(feats)
                return out
                
        else:
            feats = self.encoder(x)
            out = self.decoder(feats)
            return out