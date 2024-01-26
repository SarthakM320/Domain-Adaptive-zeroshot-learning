import builtins
import torch
from torch import nn
import numpy as np
import random
from models import CLIPVisionTransformer, CLIPResNet, CLIPTextEncoder, Decoder, SegDecoder, OverallModel
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
import yaml
from data import dataset, labels
from metrics import dice_score, precision_and_recall, intersection_over_union

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

    if args['mac'] and not args['kaggle']:
        device = 'mps'
        gpu_id = -2


    seed = args['seed']
    set_seed(seed)

    print(device)

    color_mapping = {label.color:label.id for label in labels}
    
    train_dataset = dataset(
        'dataset/cityscape_train.csv', 
        image_size = args['img_size'],
        color_mapping=color_mapping,
        kaggle = args['kaggle']
    )
    val_dataset = dataset(
        'dataset/cityscape_val.csv', 
        image_size = args['img_size'] ,  
        color_mapping=color_mapping, 
        kaggle = args['kaggle']
    )

    if args['use_gpu'] and not args['mac']:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = val_sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=train_sampler is None, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=val_sampler is None, sampler = val_sampler)

    exp=f'{args["folder"]}/'+args['exp_name']
    if gpu_id == 0:
        writer = SummaryWriter(exp)
        with open(f'{exp}/params.json', 'w') as f:
            json.dump(args, f)

    num_epochs = args['num_epochs']

    models = {}
    models['image_encoder'] = CLIPVisionTransformer(
        input_resolution = args['img_size'],
        patch_size = args['VIT']['patch_size'],
        width = args['VIT']['width'],
        layers = args['VIT']['layers'],
        heads = args['VIT']['heads'],
        output_dim = args['VIT']['output_dim'],
        drop_path_rate = args['VIT']['drop_path_rate'],
        out_indices = args['VIT']['out_indices'],
        get_embeddings = args['VIT']['get_embeddings'],
        pretrained = args['VIT']['pretrained']
    )

    models['text_encoder'] = CLIPTextEncoder(
        context_length = args['text_encoder']['context_length'],
        vocab_size = args['text_encoder']['vocab_size'],
        transformer_heads = args['text_encoder']['transformer_heads'],
        transformer_width = args['text_encoder']['transformer_width'],
        transformer_layers = args['text_encoder']['transformer_layers'],
        embed_dim = args['text_encoder']['embed_dim'],
        pretrained = args['text_encoder']['pretrained']
    )

    models['resnet'] = CLIPResNet(
        layers = args['Resnet']['layers'],
        output_dim = args['Resnet']['output_dim'],
        heads = args['Resnet']['heads'],
        input_resolution = args['img_size'],
        width = args['Resnet']['width'],
        pretrained = args['Resnet']['pretrained']
    )

    models['decoder'] = Decoder(
        output_resolution = args['Decoder']['output_resolution'],
        num_classes = args['num_classes'],
        bilinear = args['Decoder']['bilinear']
    )

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

    class_names = ['void']
    for label in labels:
        if not label.id == 0:
            class_names.append(label.name)

    
    model = OverallModel(
        image_encoder = models['image_encoder'],
        text_encoder = models['text_encoder'],
        clip_resnet = models['resnet'],
        unet_decoder = models['decoder'],
        decode_head = models['seg_decoder'],
        class_names=class_names,
        base_class = [],
        novel_class = [],
        both_class = [],
        device = device
    )

    loss_bce = nn.BCELoss()

    if args['use_gpu']:
        model = DDP(model.to(device), device_ids = [gpu_id], find_unused_parameters=True)
    model.to(device)

    # b1_seg, b1_l, feature_l, seg = model(torch.randn(1,3,512,512).to(device),torch.randn(1,3,512,512).to(device))

    optim = torch.optim.Adam(model.parameters(), lr = args['lr'])

    for epoch in range(num_epochs):
        for idx, (b1,b2,gt) in enumerate(tqdm(train_dataloader)):
            b1 = b1.to(device)
            b2 = b2.to(device)
            gt = gt.to(device)

            b1_seg, b1_l, feature_l, seg = model(b1, b2)
            # what should the threshold be
            # loss = loss_bce(b1_seg, seg['pred_masks']) + loss_bce(seg['pred_masks'], gt)
            loss = loss_bce(seg['pred_masks'], gt)
            

            optim.zero_grad()
            loss.backward()
            optim.step()

            # b1_seg = (b1_seg>args['threshold']).int()
            # seg['pred_masks'] = (seg['pred_masks']>args['threshold']).int()
            iou, _ = intersection_over_union((seg['pred_masks']>args['threshold']).int(), gt.int())
            prec, _, recall, _ = precision_and_recall((seg['pred_masks']>args['threshold']).int(), gt.int())
            dice,_ = dice_score((seg['pred_masks']>args['threshold']).int(), gt.int())

            if idx%10 == 0:
                print(loss_bce(b1_seg, seg['pred_masks']))
                print(f'PRECISION: {prec}')
                print(f'RECALL: {recall}')
                print(f'IOU: {iou}')
                print(f'DICE SCORE: {dice}')

                print(f'LOSS: {loss}')






if __name__ == "__main__":
    config = yaml.safe_load(open('trial_config.yaml', 'r'))
    main(config)