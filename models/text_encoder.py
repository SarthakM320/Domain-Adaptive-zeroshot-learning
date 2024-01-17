import torch
import clip
import torch.nn as nn
from utils_own import freeze_stages



class CLIPTextEncoder(nn.Module):
    def __init__(self, gpu_id = 0, freeze = True):
        super().__init__()
        if gpu_id == -1:
            device = 'cpu'
        else:
            device = f'cuda:{gpu_id}'
        model, _ = clip.load('ViT-B/16', device = device)
        self.vocab_size = model.vocab_size
        self.token_embedding = model.token_embedding
        self.position_embeddings = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        freeze_stages(self)
        del model

    def forward(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding 
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
