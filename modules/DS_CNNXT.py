from functools import partial
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft
from modules import *


class SEModule2d(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEModule2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GlobalFilter(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, (w//2)+1, dim, 2, dtype=torch.float32) * 0.02)
        self.h = h
        self.w = w
        
    def forward(self, x):
        x = x.permute(0,2,3,1).to(torch.float32)
        
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm='ortho')
        
        return x.permute(0,3,1,2) 


class res2conv(nn.Module):
    r""" Res2Conv
    """
    def __init__(self, dim, kernel_size, dilation, scale=8):
        super().__init__()
        width      = int(math.floor(dim / scale))
        self.nums  = scale -1
        convs      = []
        bns        = []
        num_pad    = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)
        self.act   = nn.ReLU()
        self.width = width
        
    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          # sp = self.act(sp)
          # sp = self.bns[i](sp)
          if i==0:
            x = sp
          else:
            x = torch.cat((x, sp), 1)
        x = torch.cat((x, spx[self.nums]),1)  
        return x


class LocalBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, scale=4, drop_path=0.):
        super().__init__()
        self.res2conv1 = res2conv(dim, kernel_size, dilation, scale)     
        
        self.norm = nn.BatchNorm2d(dim)   
     
        self.proj1 = nn.Conv2d(dim, 4*dim, kernel_size=1)  
        self.proj2 = nn.Conv2d(4*dim, dim, kernel_size=1)
        
        self.act   = nn.GELU()
        self.se    = SEModule2d(dim)

    def forward(self, x):
        skip = x
        
        x = self.res2conv1(x)
        x = self.norm(x)
        
        x = self.proj1(x)
        x = self.act(x) 
       
        x = self.proj2(x)

        x = skip + self.se(x)
        
        return x    


    
class GlobalBlock(nn.Module):
    """ 
     Global block: if global modules = MSA or LSTM, need to permute the dimension of input tokens
    """
    def __init__(self, dim, h=16, w=800):
        super().__init__()
        self.gf1 = GlobalFilter(dim, h=h, w=w) # Global-aware filters
        
        self.norm = nn.BatchNorm2d(dim)  

        self.proj1 = nn.Conv2d(dim, 4*dim, kernel_size=1)  
        self.proj2 = nn.Conv2d(4*dim, dim, kernel_size=1)  
        
        self.act   = nn.GELU()

    def forward(self, x):
        skip = x
        
        x = self.gf1(x) 
        x = self.norm(x) 
        
        x = self.proj1(x)
        x = self.act(x)   

        x = self.proj2(x)
      
        x = skip + x
        
        return x   


class LocalBlocks(nn.Module):
    def __init__(self, n_blocks, dim, kernel_size=3, dilation=1, scale=4, drop_path=0.):
        super().__init__()    
        self.nums   = n_blocks
        localBlocks = []
        for i in range(self.nums):
            localBlocks.append(LocalBlock(dim=dim, kernel_size=kernel_size, dilation=dilation, scale=scale, drop_path=drop_path))
        self.localBlocks = nn.ModuleList(localBlocks)
    
    def forward(self, x):
        for i in range(self.nums):
            x = self.localBlocks[i](x)
        
        return x


class GlobalBlocks(nn.Module):
    def __init__(self, n_blocks, dim, h=16, w=800):
        super().__init__()    
        self.nums   = n_blocks
        GlobalBlocks = []
        for i in range(self.nums):
            GlobalBlocks.append(GlobalBlock(dim=dim, h=h, w=w))
        self.GlobalBlocks = nn.ModuleList(GlobalBlocks)
    
    def forward(self, x):
        for i in range(self.nums):
            x = self.GlobalBlocks[i](x)
        
        return x

    
class ConvDownsampling(nn.Sequential):
    def __init__(self, inp, oup, r, bias=False):
        super().__init__()
        self.add_module('downsampling_conv', nn.Conv2d(inp, oup, kernel_size=r, stride=r, bias=bias))
        self.add_module('downsampling_norm', nn.GroupNorm(num_groups=1, num_channels=oup))

        
class FirstConvDownsampling(nn.Sequential):
    def __init__(self, inp, oup, r, bias=False):
        super().__init__()
        self.add_module('FirstConvDownsampling_conv', nn.Conv2d(inp, oup//2, kernel_size=3,stride=2,padding=1, bias=bias))
        self.add_module('FirstConvDownsampling_norm', nn.BatchNorm2d(oup//2))
        self.add_module('FirstConvDownsampling_act', nn.GELU())
        self.add_module('FirstConvDownsampling_conv2', nn.Conv2d(oup//2, oup, kernel_size=3,stride=2,padding=1, bias=bias))
        self.add_module('FirstConvDownsampling_norm2', nn.BatchNorm2d(oup))
        self.add_module('FirstConvDownsampling_act2', nn.GELU())


class dscnnxt(nn.Module):
    def __init__(self, nIn, nOut, imgH, imgW, channels=(128, 256, 512, 1024), n_blocks=[2,2,6,2], uniform_init=True):
        super().__init__()
        
        self.stem = FirstConvDownsampling(inp=nIn, oup=channels[0], r=(2,1))
        self.downsample1 = ConvDownsampling(inp=channels[0]//2, oup=channels[1]//2, r=(2,1))
        self.downsample2 = ConvDownsampling(inp=channels[1]//2, oup=channels[2]//2, r=(2,1))
        self.downsample3 = ConvDownsampling(inp=channels[2]//2, oup=channels[3]//2, r=(2,1))

        # local branch
        # self.llayer1 = LocalBlocks(n_blocks[0], channels[0]//2, kernel_size=3, scale=4, dilation=1) #默认8 8 8 C=1024; 尝试4 6 8 C=960
        self.llayer1 = LocalBlock(channels[0]//2, kernel_size=3, scale=4, dilation=1) #默认8 8 8 C=1024; 尝试4 6 8 C=960
        self.llayer2 = LocalBlock(channels[1]//2, kernel_size=3, scale=4, dilation=1)
        self.llayer3 = LocalBlock(channels[2]//2, kernel_size=3, scale=4, dilation=1)
        self.llayer4 = LocalBlock(channels[3]//2, kernel_size=3, scale=4, dilation=1)
        
        #global branch
        # self.glayer1 = GlobalBlocks(n_blocks[0], channels[0]//2, h=imgH//4,  w=imgW//4)
        self.glayer1 = GlobalBlock(channels[0]//2, h=imgH//4,  w=imgW//4)
        self.glayer2 = GlobalBlock(channels[1]//2, h=imgH//8,  w=imgW//4)
        self.glayer3 = GlobalBlock(channels[2]//2, h=imgH//16, w=imgW//4)
        self.glayer4 = GlobalBlock(channels[3]//2, h=imgH//32, w=imgW//4)
        
        self.last_conv = nn.Conv2d(
                in_channels=channels[-1],
                out_channels=nOut,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        self.dropout = nn.Dropout(0.1)
        self.uniform_init = uniform_init
        
    def forward(self, x):
        # stage 1:
        x = self.stem(x)
        
        lx, gx = torch.chunk(x, 2, dim=1)
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1(gx)

        
        mix1 = lx1 + gx1
        mix1 = self.downsample1(mix1)
        
        lx2 = self.llayer2(mix1)
        gx2 = self.glayer2(mix1)
        
        mix2 = lx2 + gx2
        mix2 = self.downsample2(mix2)
        
        lx3 = self.llayer3(mix2)
        gx3 = self.glayer3(mix2)
        
        mix3 = lx3 + gx3
        mix3 = self.downsample3(mix3)
        
        lx4 = self.llayer4(mix3)
        gx4 = self.glayer4(mix3)

        mix_x = torch.cat((lx4, gx4), dim=1)
        
        x = self.last_conv(mix_x)
        x = self.dropout(x)
        
        return x

  
    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)     
      