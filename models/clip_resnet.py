from collections import OrderedDict
import torch
import clip
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
from models.utils_own import *
from timm.models.layers import trunc_normal_
import math


class CLIPResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, pretrained = None):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        # print(layers)
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.layers = [self.layer1,self.layer2,self.layer3,self.layer4]

        self.embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, self.embed_dim, heads, output_dim)

        if self.pretrained:
            self.init_weights()
            freeze_stages(self)
            

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            state_dict = {k.replace('visual.', ''):v for k,v in checkpoint.items() if k.startswith('visual')}
            if 'attnpool.positional_embedding' in state_dict.keys():
                if state_dict['attnpool.positional_embedding'].shape != self.attnpool.positional_embedding.shape:
                    print(f'Resize the pos_embed shape from {state_dict["attnpool.positional_embedding"].shape} to {self.attnpool.positional_embedding.shape}')
                    cls_pos = state_dict["attnpool.positional_embedding"][0:1, :]
                    spatial_pos = F.interpolate(state_dict['attnpool.positional_embedding'][1:].reshape(1,7,7,self.embed_dim).permute(0,3,1,2), size=(self.input_resolution//32,self.input_resolution//32), mode='bilinear')
                spatial_pos = spatial_pos.reshape(self.embed_dim, 256).permute(1, 0)
                positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                state_dict['attnpool.positional_embedding'] = positional_embedding
                assert self.attnpool.positional_embedding.shape == state_dict['attnpool.positional_embedding'].shape
            u,w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in resnet')

    def get_features(self, image, layers='all'):
        feats = []
        def hook(module, input, output):
            feats.append(output)
            
        if isinstance(layers, str):
            if layers.lower() == 'all':
                ids = [l.register_forward_hook(hook) for l in self.layers]
            else:
                ids = [self.layers[int(layers)].register_forward_hook(hook)]
        elif isinstance(layers, list):
            ids = [self.layers[l].register_forward_hook(hook) for l in layers]


        _ = self(image)

        for id in ids:
            id.remove()

        return feats