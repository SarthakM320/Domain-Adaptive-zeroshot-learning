import builtins
import torch
from torch import nn
import numpy as np
import random
from models import CLIPViT, CLIPResnet, CLIPTextEncoder, Decoder, SegDecoder
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm 
import json
import warnings
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import random
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main(args):
    if args['use_gpu']:
        ddp_setup()
        gpu_id = int(os.environ["LOCAL_RANK"])
        device = gpu_id
    else:
        gpu_id = -1
        device = 'cpu'


    args = vars(args)
    seed = args['seed']
    set_seed(seed)


    exp=f'{args["folder"]}/'+args['exp_name']
    if gpu_id == 0:
        writer = SummaryWriter(exp)
        with open(f'{exp}/params.json', 'w') as f:
            json.dump(args, f)

    num_epochs = args['num_epochs']

    models = {}
    models['image_encoder'] = CLIPViT(
        model_type = args['VIT']['model_type'],
        gpu_id = gpu_id,
        freeze = args['VIT']['freeze']
    )
    models['text_encoder'] = CLIPTextEncoder(
        gpu_id=gpu_id,
        freeze = args['text_encoder']['freeze']
    )
    models['resnet'] = CLIPResnet(
        model_type = args['Resnet']['model_type'],
        gpu_id=gpu_id,
        freeze=args['Resnet']['freeze']
    )
    models['decoder'] = Decoder(args['num_classes'])

    seen_idx = all_idx = None
    models['seg_decoder'] = SegDecoder(
        args['img_size'], 
        args['SegDecoder']['in_channels'], 
        seen_idx, 
        all_idx, 
        args['num_classes'], 
        args['SegDecoder']['embed_dim'], 
        args['SegDecoder']['num_layers'],
        args['SegDecoder']['num_heads'],
        args['SegDecoder']['use_stages'], 
        args['SegDecoder']['use_proj']
    )

    if args['use_gpu']:
        for model_name in models.keys():
            models[model_name].to(device)


# if __name__ == "__main__":
#     pass