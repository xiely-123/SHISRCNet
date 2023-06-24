import math
import torch.nn as nn
import basicblock as B
import torch
import math
import torch.nn.functional as F
import torchvision
import SKnet as SK
import profile
'''
# ===================
# srresnet
# ===================
'''

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):####16
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=15, stride=1, padding=7)####kernel_size=7
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        #print(out.size())
        out = self.spatial_attention(out) * out
        return out


"""
features = torch.rand((8, 64, 192, 192))
attention = CBAM(64)
result = attention(features)

print(result.size()) 
"""

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),#groups = in_channels 
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(CBAM(in_channels))#ASPPPooling(in_channels, out_channels)selfAttention(64,192*192,192*192)
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
            #nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
"""
aspp = ASPP(64,[2,4,8])
x = torch.rand(2,64,192,192)
print(aspp(x).shape)

"""
class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class MFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(MFEblock, self).__init__()
        out_channels = in_channels
        # modules = []
        # modules.append(nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),#groups = in_channels , bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
            #nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.softmax_1 = nn.Sigmoid()
        self.SE1 = oneConv(in_channels,in_channels,1,0,1)
        self.SE2 = oneConv(in_channels,in_channels,1,0,1)
        self.SE3 = oneConv(in_channels,in_channels,1,0,1)
        self.SE4 = oneConv(in_channels,in_channels,1,0,1)
    def forward(self, x):
        y0 = self.layer1(x)
        y1 = self.layer2(y0+x)
        y2 = self.layer3(y1+x)
        y3 = self.layer4(y2+x)
        #res = torch.cat([y0,y1,y2,y3], dim=1)
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.project(x_att+x) 



# aspp = MFEblock(64,[2,4,8])
# x = torch.rand(2,64,192,192)
# print(aspp(x).shape)




class SRResNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=3, upscale=4, act_mode='L', upsample_mode='pixelshuffle'):##act_mode='L'3;2  128
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [MFEblock(nc,[2,4,8]) for _ in range(nb)]
        #m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(15)]
        #m_body.append([MFEblock(nc,[2,4,8]) for _ in range(1)])
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.backbone = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)))
        
        self.up =B.sequential(*m_uper, m_tail)
    def forward(self, x):
        SR_feature = self.backbone(x)
        SR = self.up(SR_feature)
        return SR_feature,SR

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class CSFblock(nn.Module):
    ###联合网络
    def __init__(self, in_channels, channels_1, strides):
        super().__init__()
        #self.layer = nn.Conv1d(in_channels, 512, kernel_size = 1, padding = 0, dilation = 1)
        self.Up = nn.Sequential(
            #nn.MaxPool2d(kernel_size = int(strides*2+1), stride = strides, padding = strides),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size = 2, stride = strides, padding = 0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),            
        )
        self.Fgp = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Sequential(
            oneConv(in_channels,channels_1,1,0,1),
            oneConv(channels_1,in_channels,1,0,1),
                      
        )
        self.SE1 = oneConv(in_channels,in_channels,1,0,1)
        self.SE2 = oneConv(in_channels,in_channels,1,0,1)
        self.softmax = nn.Softmax(dim = 2)
    def forward(self, x_h, x_l):
        
        x1 = x_h
        x2 = self.Up(x_l)
        
        x_f = x1+x2
        #print(x_f.size())
        Fgp = self.Fgp(x_f)
        #print(Fgp.size())
        x_se = self.layer1(Fgp)
        x_se1 = self.SE1(x_se)
        x_se2 = self.SE2(x_se)
        x_se = torch.cat([x_se1,x_se2],2)
        x_se = self.softmax(x_se)
        att_3 = torch.unsqueeze(x_se[:,:,0],2)
        att_5 = torch.unsqueeze(x_se[:,:,1],2)
        x1 = att_3*x1
        x2 = att_5*x2
        x_all = x1+x2
        return x_all   



"""
x = torch.rand(1,32,100,100) 
x1 = torch.rand(1,32,200,200)
net = CSFblock(32,16,2)
print(net)
y = net(x1,x)     
print(y.size())

"""


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.CSF1 = CSFblock(128,32,2)
        self.CSF2 = CSFblock(128,32,2)
        self.GAP =  nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        x3 = x3
        x2 = self.CSF1(x2,x3)
        x1 = self.CSF2(x1,x2)
        x3 = self.GAP(x3).squeeze(-1).squeeze(-1)
        x2 = self.GAP(x2).squeeze(-1).squeeze(-1)
        x1 = self.GAP(x1).squeeze(-1).squeeze(-1)   
        results = torch.cat([x3,x2,x1],1)
        return results

class CF(nn.Module):
    def __init__(self, class_num):
        super(CF, self).__init__()
        self.backbone = SK.SKNet26(2)
        self.FPN = FPN()
        self.fc = nn.Sequential(  
                  nn.Linear(384, 256),
                  nn.Dropout(0.1),
                  nn.ReLU(),
                  nn.Linear(256, 2),
                  )       
    def forward(self, x):
        feature = self.backbone(x)
        #print(feature.size())
        results = self.fc(self.FPN(feature))
        return results


"""
x = torch.rand(3,3,384,384) 
net = CF(2)
#print(net)
y = net(x)     
print(y.size())
"""


class SIHSRCNet(nn.Module):
    def __init__(self, ):
        super(SIHSRCNet, self).__init__()
        self.SRNet = SRResNet()
        self.classNet = CF(2)
    def forward(self, x):
        _,SR = self.SRNet(x)
        results = self.classNet(SR)
        return SR,results
"""
x = torch.rand(10,3,192,192)
#x1 =  torch.rand(1,64,192,192)
net = SIHSRCNet()
#print(net)
y,y1= net(x)     
print(y.size(),y1.size())
"""
# from ptflops import get_model_complexity_info
# if __name__ == '__main__':
    # model = SRResNet()
    # flops, params = get_model_complexity_info(model, (3, 192, 192), as_strings=True, print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)