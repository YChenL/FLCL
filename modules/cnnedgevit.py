import torch
import torch.nn as nn

edgevit_configs = {
    'XXS': {
        'channels': (36, 72, 144, 288),
        'blocks': (1, 1, 3, 2),
        'heads': (1, 2, 4, 8)
    }
    ,
    'XS': {
        'channels': (48, 96, 240, 384),
        'blocks': (1, 1, 2, 2),
        'heads': (1, 2, 4, 8)
    }
    ,
    'S': {
        'channels': (48, 96, 240, 384),
        'blocks': (1, 2, 3, 2),
        'heads': (1, 2, 4, 8)
    }
}

HYPERPARAMETERS = {
    'r': (4, 2, 2, 1)
}


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        return x + self.module(x)


class ConditionalPositionalEncoding(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        self.add_module('conditional_ositional_encoding', nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False))


class MLP(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        expansion = 4
        self.add_module('mlp_layer_0', nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False))
        self.add_module('mlp_act', nn.GELU())
        self.add_module('mlp_layer_1', nn.Conv2d(channels*expansion, channels, kernel_size=1, bias=False))


class LocalAggModule(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        self.add_module('pointwise_prenorm_0', nn.BatchNorm2d(channels))
        self.add_module('pointwise_conv_0', nn.Conv2d(channels, channels, kernel_size=1, bias=False))
        self.add_module('depthwise_conv', nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False))
        self.add_module('pointwise_prenorm_1', nn.BatchNorm2d(channels))
        self.add_module('pointwise_conv_1', nn.Conv2d(channels, channels, kernel_size=1, bias=False))


class GlobalSparseAttetionModule(nn.Module):
    def __init__(self, channels, r, heads):
        super().__init__()
        self.head_dim = channels//heads
        self.scale = self.head_dim**-0.5
        self.num_heads = heads

        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.local_prop = nn.ConvTranspose2d(channels, channels, kernel_size=r, stride=r, groups=channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H*W).split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        x = self.local_prop(x)
        x = self.norm(x)
        x = self.proj(x)

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


class cnnEdgeViT(nn.Module):
    def __init__(self, nIn, nOut, channels=(48, 96, 256), blocks=(1, 1, 2), heads=(1, 4, 8), r=[4, (2,1), 1], num_classes=1000, distillation=False):
        super().__init__()
        self.distillation = distillation
        
        l = []
        in_channels = nIn
        for stage_id, (num_channels, num_blocks, num_heads, sample_ratio) in enumerate(zip(channels, blocks, heads, r)):
            if stage_id == 0:
                l.append(FirstConvDownsampling(inp=in_channels, oup=num_channels, r=(2,1)))
            elif stage_id == 1:
                l.append(ConvDownsampling(inp=in_channels, oup=num_channels, r=(2,2)))
            else:
                l.append(ConvDownsampling(inp=in_channels, oup=num_channels, r=(2,1)))
                #  if stage_id == 0 else 2))
            for _ in range(num_blocks):
                l.append(Residual(ConditionalPositionalEncoding(num_channels)))
                l.append(Residual(GlobalSparseAttetionModule(channels=num_channels, r=sample_ratio, heads=num_heads)))
                l.append(Residual(MLP(num_channels)))
            
            in_channels = num_channels
        
        self.main_body = nn.Sequential(*l)

        self.last_conv = nn.Conv2d(
                in_channels=256,
                out_channels=nOut,
                kernel_size=(2,1),
                stride=1,
                padding=0,
                bias=False
            )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.main_body(x)
        x = self.last_conv(x)
        x = self.dropout(x)

        
        return x


def EdgeViT_XXS(pretrained=False):
    model = cnnEdgeViT(**edgevit_configs['XXS'])
    
    if pretrained:
        raise NotImplementedError
    
    return model

def EdgeViT_XS(pretrained=False):
    model = cnnEdgeViT(**edgevit_configs['XS'])
    
    if pretrained:
        raise NotImplementedError
    
    return model

def EdgeViT_S(pretrained=False):
    model = cnnEdgeViT(**edgevit_configs['S'])
    
    if pretrained:
        raise NotImplementedError
    
    return model


from torchsummary import summary
if __name__ == '__main__':
    # img = torch.rand([1, 3, 32, 640]).to("cuda")
    svtr = cnnEdgeViT()
    print(svtr)
    summary(svtr, input_size=[(3, 32, 1600)], batch_size=1, device="cpu")
    from torchstat import stat
    # stat(svtr, (3, 32, 800))
    # summary(svtr, input_size=[(3, 32, 1600)], batch_size=1, device="cpu")
    # summary(svtr, input_size=[(3, 32, 400)], batch_size=1, device="cpu")
    stat(svtr, (3, 32, 1600))
    # stat(svtr, (3, 32, 176))
    # stat(svtr, (3, 32, 144))
    # # print(x[0].to("cpu").shape)

    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(svtr, (3, 32, 32), as_strings=True, print_per_layer_stat=True) #不用写batch_size大小，默认batch_size=1
    # print('Flops:  ' + flops)
    # print('Params: ' + params)