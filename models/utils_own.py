import torch
from torch import nn
from torch.nn import functional as F

def bilinear_upsample(x, target_shape):
    return F.interpolate(x, size = target_shape, mode = 'bilinear')

def freeze_stages(model, exclude_key=None):
    """Freeze stages param and norm stats."""
    for n, m in model.named_parameters():
        if exclude_key:
            if isinstance(exclude_key, str):
                if not exclude_key in n:
                    m.requires_grad = False
            elif isinstance(exclude_key, list):
                count = 0
                for i in range(len(exclude_key)):
                    i_layer = str(exclude_key[i])
                    if i_layer in n:
                        count += 1
                if count == 0:
                    m.requires_grad = False
                elif count>0:
                    print('Finetune layer in backbone:', n)
            else:
                assert AttributeError("Dont support the type of exclude_key!")
        else:
            m.requires_grad = False

def stylize(feats_b1, feats_b2):
    stylized_b1 = []
    for b1, b2 in zip(feats_b1, feats_b2):
        b1_mean, b1_std = compute_mean_std(b1)
        b2_mean, b2_std = compute_mean_std(b1)
        stylized_b1.append(b2_std*((b1-b1_mean)/b1_std)+b2_mean)
    
    return stylized_b1

def compute_mean_std(feats, eps=1e-8):
    n,c,_,_ = feats.shape
    feats = feats.view(n, c, -1)
    mean = torch.mean(feats, dim=-1).view(n, c, 1, 1)
    std = torch.std(feats, dim=-1).view(n, c, 1, 1) + eps

    return mean, std
    
def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
# self.out_indices = [3,5,7,11]
# features = []
# outs = []
# for i, blk in enumerate(self.transformer.resblocks):
#     x = blk(x)
#     if len(self.out_indices) > 1:
#         if i in self.out_indices:
#             xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
#             features.append(xp.contiguous())
# x = x.permute(1, 0, 2)
# x = self.ln_post(x)
# x = x @ self.proj
# global_embedding = x[:, 0]
# visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)

# if len(self.out_indices) == 1:
#     visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True)
#     features.append(visual_embedding)

# outs.append(tuple(features))

# global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)
# outs.append(global_embedding) 

# return outs    