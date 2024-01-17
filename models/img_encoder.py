import torch
import clip
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
from models.utils_own import freeze_stages, trunc_normal_init, constant_init
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from timm.models.layers import trunc_normal_
import math
from mmseg.models.losses import accuracy

class CLIPImageEncoder(nn.Module):
    def __init__(self, image_encoder, freeze = True, **kwargs):
        super().__init__(**kwargs)

        self.image_encoder = image_encoder
        self.image_encoder_named_modules = dict(image_encoder.named_modules())
        if freeze:
            freeze_stages(self.image_encoder)

    def forward(self, image):
        return self.image_encoder(image)
    
    def get_features(self, image, layers='all'):
        feats = []
        def hook(module, input, output):
            feats.append(output)
            
        if isinstance(layers, str):
            if layers.lower() == 'all':
                ids = [self.image_encoder_named_modules[l].register_forward_hook(hook) for l in self.all_layers]
            else:
                ids = [self.image_encoder_named_modules[layers].register_forward_hook(hook)]
        elif isinstance(layers, list):
            ids = [self.image_encoder_named_modules[l].register_forward_hook(hook) for l in layers]
    

        _ = self.image_encoder(image)

        for id in ids:
            id.remove()

        return feats

class CLIPViT(CLIPImageEncoder):
    def __init__(self, model_type = 'ViT-B/16', gpu_id= 0, freeze = True):
        if gpu_id == -1:
            device = 'cpu'
        else:
            device = f'cuda:{gpu_id}'
        
        model, _ = clip.load(model_type, device = device)
        self.patch_size = int(model_type.split('/')[-1])
        super().__init__(model.visual, freeze = freeze)

        self.all_layers = [3,5,7,11]


    def get_features(self, image, layers=[11]):
        B, C, H, W = image.shape
        H = H//self.patch_size
        W = W//self.patch_size

        feats = []
        def hook(module, input, output):
            xp = output.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
            feats.append(xp.contiguous())
        
        transformer_feats = {}
        def transformer_hook(module, input, output):
            transformer_feats['transformer_out'] = output

        if isinstance(layers, str):
            if layers.lower() == 'all':
                ids = [block.register_forward_hook(hook) for block in self.image_encoder.transformer.resblocks]
            else:
                try:
                    layers = int(layers)
                    ids = [self.image_encoder.transformer.resblocks[layers].register_forward_hook(hook)]
                except:
                    raise AssertionError('layer number not in correct format')
        elif isinstance(layers, int):
            ids = [self.image_encoder.transformer.resblocks[layers].register_forward_hook(hook)]
        elif isinstance(layers, list):
            ids = [self.image_encoder.transformer.resblocks[int(layer)].register_forward_hook(hook) for layer in layers]

        id = self.image_encoder.transformer.register_forward_hook(transformer_hook)
        
        _ = self.image_encoder(image)

        out = transformer_feats['transformer_out'].permute(1,0,2)
        x = self.image_encoder.ln_post(out) @ self.image_encoder.proj
        global_embedding = x[:, 0]
        visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)

        id.remove()
        for i in ids:
            i.remove()

        if len(feats) == 1:
            visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True)
            feats.append(visual_embedding)

        out = [tuple(feats)]
        global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)
        out.append(global_embedding) 

        return out

class CLIPResnet(CLIPImageEncoder):
    def __init__(self, model_type = 'RN50', gpu_id = 0, freeze = True):
        if gpu_id == -1:
            device = 'cpu'
        else:
            device = f'cuda:{gpu_id}'

        model, _ = clip.load(model_type, device = device)
        super().__init__(model.visual, freeze = freeze)
    
        self.all_layers = ['layer1','layer2','layer3','layer4']
