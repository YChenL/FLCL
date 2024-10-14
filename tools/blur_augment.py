from io import BytesIO
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from skimage.filters import gaussian
from wand.image import Image as WandImage


def disk(radius, alias_blur=0.1, kernel_size=10, dtype=np.float32):
    if radius <= kernel_size:
        coords = np.arange(-kernel_size, kernel_size + 1)
        ksize = (3, 3)
    else:
        coords = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(coords, coords)
    aliased_disk = np.asarray((x ** 2 + y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


class GaussianBlur(nn.Module):
    def __init__(self, rng=None):
        super(GaussianBlur, self).__init__()
        self.rng    = np.random.default_rng() if rng is None else rng
        self.sigmas = [0.5, 0.8, 1.2] #  [0.5, 4, 8] # provide 5 level blur
        
    def forward(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img
        "img: Tensor, minibatch data (b, c, w, h)"
        
        _, _, w, h = img.size()
        ksize = int(min(w, h) / 2) // 4
        ksize = (ksize * 2) + 1
        kernel = (ksize, ksize)

        if mag < 0 or mag >= len(self.sigmas):
            index = self.rng.integers(0, len(self.sigmas))
        else:
            index = mag
            
        sigma = self.sigmas[index]
        return transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)
        
        
class DefocusBlur(nn.Module):
    def __init__(self, device, nIn=1, ks=21, rng=None):
        super(DefocusBlur, self).__init__()
        self.device = device
        self.rng = np.random.default_rng() if rng is None else rng
        self.c   = [(1, 0.025), (2, 0.05), (2, 0.1)] # , (8, 0.5), (10, 0.5)]
        self.channels = nIn
        self.ks    = ks

        # defualt padding model in cv2.filter2d is "reflect padding"
        self.conv  = nn.Conv2d(in_channels=nIn, out_channels=nIn, kernel_size=ks,  
                                stride=1, padding=(ks-1)//2, padding_mode='reflect', 
                                groups=nIn, bias=False).to(device) 
            
    def forward(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img
        "img: Tensor, minibatch data (b, c, w, h)"

        img = img / 255.
        if mag < 0 or mag >= len(self.c):
            index = self.rng.integers(0, len(self.c))
        else:
            index = mag
        c = self.c[index]

        # generate 3d kernal, (c, 1, ks, ks)
        kernel = np.repeat(disk(radius=c[0], alias_blur=c[1], kernel_size=(self.ks-1)//2).reshape(1,1,self.ks,self.ks), self.channels, axis=0)
        kernel = torch.tensor(kernel).to(self.device) 
        self.conv.weight = nn.Parameter(kernel)
        with torch.no_grad():
            img = self.conv(img)

        img = torch.floor((torch.clamp(img, 0, 1) * 255))
        
        return img


class MotionBlur(nn.Module):
    def __init__(self, device, nIn=1, rng=None):
        super(MotionBlur, self).__init__()
        self.device = device
        self.rng = np.random.default_rng() if rng is None else rng
        self.c   = [4, 5, 6] # , 22, 24] 
        self.channels = nIn

        self.conv  = nn.Conv2d(in_channels=nIn, out_channels=nIn, kernel_size=1,  
                                stride=1, groups=nIn, bias=False).to(device) 
     
    def forward(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img
        "img: Tensor, minibatch data (b, c, w, h)"
        
        if mag < 0 or mag >= len(self.c):
            index = self.rng.integers(0, len(self.c))
        else:
            index = mag
        c = self.c[index]
        
        # generate kernel
        kern = np.ones((1, c), np.float32)
        angle = self.rng.uniform(-45, 45)
        angle = -np.pi*angle/180
        cos, sin = np.cos(angle), np.sin(angle)
        A = np.float32([[cos, -sin, 0], [sin, cos, 0]])
        sz2 = c // 2
        A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((c-1)*0.5, 0))
        
        kern = cv2.warpAffine(kern, A, (c, c), flags=cv2.INTER_CUBIC)
        kern /= np.sum(kern)
        kern = np.repeat(kern.reshape(1,1,c,c), self.channels, axis=0)  
        kern = torch.tensor(kern).to(self.device) 
        
        # reflect padding for convolution
        img = F.pad(img/255., (sz2, sz2-1, sz2-1, sz2), 'reflect')
        self.conv.weight = nn.Parameter(kern)
        
        with torch.no_grad():
            img = self.conv(img)
        
        img = torch.floor((torch.clamp(img, 0, 1) * 255))
            
        return img


class GlassBlur:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.c   = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)] # [severity - 1]

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img
        "img: Tensor, minibatch data (b, c, w, h)"
        
        _, _, h, w = img.size()
        if mag < 0 or mag >= len(self.c):
            index = self.rng.integers(0, len(self.c))
        else:
            index = mag

        c = self.c[index]
        
        # generate kernel
        ksize = int(min(w, h) / 2) // 4
        ksize = (ksize * 2) + 1
        kernel = (ksize, ksize)
        gaussian = transforms.GaussianBlur(kernel_size=kernel, sigma=c[0])
        img = torch.floor(gaussian(img / 255.) * 255)
        
        # locally shuffle pixels
        for i in range(c[2]):
            for y in range(h - c[1], c[1], -1):
                for x in range(w - c[1], c[1], -1):
                    dx, dy = self.rng.integers(-c[1], c[1], size=(2,))
                    y_prime, x_prime = y + dy, x + dx
                    # swap
                    img[:, :, y, x], img[:, :, y_prime, x_prime] = img[:, :, y_prime, x_prime], img[:, :, y, x]

        img = torch.floor(torch.clamp(gaussian(img / 255.), 0, 1) * 255)

        return img


class ZoomBlur:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.c   = [np.arange(1, 1.11, .01),
                    np.arange(1, 1.16, .01),
                    np.arange(1, 1.21, .02),
                    np.arange(1, 1.26, .02),
                    np.arange(1, 1.31, .03)]
        
    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        _, _, h, w = img.size()
        crop = transforms.CenterCrop((w, h))

        if mag < 0 or mag >= len(self.c):
            index = self.rng.integers(0, len(self.c))
        else:
            index = mag

        c = self.c[index]

        uint8_img = img
        img = (img / 255.)

        out = torch.zeros(img.size()) #.cuda()
        
        for zoom_factor in c:
            zw = int(w * zoom_factor)
            zh = int(h * zoom_factor)
            zoom_img = F.interpolate(uint8_img, size=(zh, zw), mode='bicubic')     

            x1 = (zw - w) // 2
            y1 = (zh - h) // 2
            x2 = x1 + w
            y2 = y1 + h

            zoom_img = zoom_img[..., y1:y2, x1:x2] #crop
            # zoom_img = crop(zoom_img)
            
            out += zoom_img / 255.
        
        # hook
        # img_hook = img

        img = (img + out) / (len(c) + 1)
        img = torch.floor(torch.clamp(img, 0, 1) * 255.)
    
        return img
