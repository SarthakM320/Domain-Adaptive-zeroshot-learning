from models.utils_own import freeze_stages, stylize
import torch
from torch import nn
from clip import tokenize
import numpy as np



class Model(nn.Module):
    def __init__(
        self, 
        image_encoder, 
        text_encoder, 
        clip_resnet,
        unet_decoder, 
        decode_head,
        class_names,
        base_class,
        novel_class,
        both_class,
        gpu_id,
        training = False,
        self_training = False,
        load_text_embedding = None,
    ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.clip_resnet = clip_resnet
        self.decode_head = decode_head
        self.unet_decoder = unet_decoder
        self.class_names = class_names
        self.gpu_id = gpu_id
        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)
        self.self_training = self_training
        self.load_text_embedding = load_text_embedding
        self.training = training

        if not self.load_text_embedding:
            self.texts = torch.cat([tokenize(f"a photo of a {c}") for c in self.class_names]) 
        # TODO write the else part
        
        


    def forward(self, image_b1, image_b2):
        b1_resnet_features = self.clip_resnet.get_features(image_b1)
        b2_resnet_features = self.clip_resnet.get_features(image_b2)
        b1_stylized = stylize(b1_resnet_features, b2_resnet_features)
        
        # pass b1_stylized to unet_decoder
        # which feature map to get from the middle
        b1_seg = self.unet_decoder(b1_stylized)

        text_features = self.get_text_embeddings(self.texts)
        #which image to pass b1 or b2
        image_features = self.get_image_features(image_b1, model = 'vit')






    def get_image_features(self, images, model='resnet'):
        if model == 'resnet':
            return self.clip_resnet.get_features(images)
        else:
            return self.image_encoder.get_features(images)
    
    def get_text_embeddings(self, text):
        text_embeddings = self.text_encoder(text)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings