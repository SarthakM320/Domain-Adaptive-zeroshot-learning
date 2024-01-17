from img_encoder import CLIPViT, CLIPResnet
from models.utils_own import freeze_stages
from text_encoder import CLIPTextEncoder
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
        
        



    def forward(self):
        pass

    def get_image_features(self, images, model='resnet'):
        if model == 'resnet':
            return self.clip_resnet.get_features(images)
        else:
            return self.image_encoder.get_features(images)
    
    def get_text_embeddings(self, text):
        text_embeddings = self.text_encoder(text)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings