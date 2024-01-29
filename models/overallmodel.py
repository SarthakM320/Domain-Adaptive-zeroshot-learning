import torch
from torch import nn
from clip import tokenize
import numpy as np
from torch.nn import functional as F
from models import *
from os.path import join

class OverallModel(nn.Module):
    def __init__(
        self, 
        image_encoder: CLIPVisionTransformer, 
        text_encoder: CLIPTextEncoder, 
        clip_resnet: CLIPResNet,
        unet_decoder: Decoder, 
        decode_head: SegDecoder,
        class_names,
        base_class,
        novel_class,
        both_class,
        device,
        training = False,
        self_training = False,
        load_text_embedding = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.clip_resnet = clip_resnet
        self.decode_head = decode_head
        self.unet_decoder = unet_decoder
        # self.class_names = class_names
        self.class_names = class_names
        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)
        self.self_training = self_training
        self.load_text_embedding = load_text_embedding
        self.training = training
        self.device = device

        if not self.load_text_embedding:
            self.texts = torch.cat([tokenize(f"a photo of a {c}") for c in self.class_names]).to(device)
        # TODO write the else part
        
        self.conv = nn.Conv2d(3,3, kernel_size = 1, stride = 1, padding = 0)
        


    def forward(self, image_b1, image_b2):
        b,c,h,w = image_b1.shape
        b1_resnet_features = self.clip_resnet.get_features(image_b1)
        b2_resnet_features = self.clip_resnet.get_features(image_b2)
        b1_stylized = stylize(b1_resnet_features, b2_resnet_features)
        b1_stylized.reverse()
        
        # pass b1_stylized to unet_decoder
        # which feature map to get from the middle
        b1_seg, b1_l = self.unet_decoder(b1_stylized)

        text_features = self.get_text_embeddings(self.texts)
        #which image to pass b1 or b2
        image_features = self.get_image_features(image_b1, model = 'vit')
        feature_l = self.conv(image_features[0][0].reshape(b,c,h,w))
        seg = self.decode_head([image_features, text_features])

        # b1_seg = F.sigmoid(b1_seg)
        # seg['pred_masks'] = F.sigmoid(seg['pred_masks'])
        return b1_seg, b1_l, feature_l, seg

    def forward_test(self, image_b2):
        b,c,h,w = image_b2.shape
        b2_resnet_features = self.clip_resnet.get_features(image_b2)
        b2_resnet_features.reverse()
        b2_seg, b2_l = self.unet_decoder(b2_resnet_features)

        # image_features = self.get_image_features(image_b2, model = 'vit')
        # feature_l = self.conv(image_features[0][0].reshape(b,c,h,w))
        # seg = self.decode_head([image_features, text_features])

        return b2_seg


    def save_model(self, path, epoch,latest = True):
        ckpt = {
            'epoch':epoch, 
            'unet_decoder':self.unet_decoder.state_dict(),
            'seg_decoder':self.decode_head.state_dict(),
        }
        if latest:
            torch.save(ckpt, join(path, 'latest.pth'))
        else:
            torch.save(ckpt, join(path, 'best.pth'))

    def load_model(self, path, latest = True):
        if latest:
            ckpt = torch.load(join(path, 'latest.pth'))
        else:
            ckpt = torch.load(join(path, 'best.pth'))

        self.unet_decoder.load_state_dict(ckpt['unet_decoder'])
        self.decode_head.load_state_dict(ckpt['seg_decoder'])
        print('Model Loaded')
        return ckpt['epoch']
        



    def get_image_features(self, images, model='resnet'):
        if model == 'resnet':
            return self.clip_resnet.get_features(images)
        else:
            return self.image_encoder(images)
    
    def get_text_embeddings(self, text):
        text_embeddings = self.text_encoder(text)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings