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
from metrics import dice_score, precision_and_recall, intersection_over_union, IoU

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
        gpu_id = 0


    seed = args['seed']
    set_seed(seed)

    print(device)

    
    train_dataset = dataset(
        'dataset/cityscape_train.csv', 
        image_size = args['img_size'],
        num_classes = args['num_classes'],
        kaggle = args['kaggle']
    )
    val_dataset = dataset(
        'dataset/cityscape_val.csv', 
        image_size = args['img_size'],
        num_classes = args['num_classes'],
        kaggle = args['kaggle']
    )

    if args['use_gpu']:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = val_sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], num_workers = 16,shuffle=train_sampler is None, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers = 8,shuffle=val_sampler is None, sampler = val_sampler)

    exp=f'{args["folder"]}/'+args['exp_name']
    if gpu_id == 0:
        writer = SummaryWriter(exp)
        with open(f'{exp}/params.json', 'w') as f:
            json.dump(args, f)
    
    if gpu_id != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    num_epochs = args['num_epochs']

    IOU = IoU(args['threshold'])

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
        if not label.id == 0 and not label.ignoreInEval:
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
    cos = nn.CosineSimilarity(dim = -1)

    if args['use_gpu']:
        model = DDP(model.to(device), device_ids = [device], find_unused_parameters=True)
    model.to(device)

    # b1_seg, b1_l, feature_l, seg = model(torch.randn(1,3,512,512).to(device),torch.randn(1,3,512,512).to(device))

    optim = torch.optim.Adam(model.parameters(), lr = args['lr'])
    
    step_train = 0
    step_val = 0
    best_iou = 0
    running_iou_train_mean = []
    for epoch in range(num_epochs):

        model.train()
        for idx, (b1,b2,gt) in enumerate(tqdm(train_dataloader)):

            
            b1 = b1.to(device)
            b2 = b2.to(device)
            gt = gt.to(device)

            b1_seg, b1_l, feature_l, seg = model(b1, b2)
            # what should the threshold be
            
            loss_cos = (1-cos(b1_l.flatten(-2), feature_l.flatten(-2))).mean()*args['cos_loss_weight']
            loss_b1_seg = loss_bce(b1_seg, seg['pred_masks'])*args['b1_seg_loss_weight']
            loss_seg = loss_bce(seg['pred_masks'], gt)*args['seg_loss_weight']
            loss = loss_b1_seg + loss_seg + loss_cos
            # loss = loss_bce(seg['pred_masks'], gt)
            

            optim.zero_grad()
            loss.backward()
            optim.step()

            precision_seg = []
            recall_seg = []
            iou_seg = []
            acc_seg = []

            for i in range(len(class_names)):
                acc,recall,prec,iou = IOU(gt[:,i,:,:], seg['pred_masks'].softmax(dim=1)[:,i,:,:])
                precision_seg.append(prec)
                recall_seg.append(recall)
                acc_seg.append(acc)
                iou_seg.append(iou)

            precision_b1_seg = []
            recall_b1_seg = []
            iou_b1_seg = []
            acc_b1_seg = []

            for i in range(len(class_names)):
                acc,recall,prec,iou = IOU(gt[:,i,:,:], b1_seg.softmax(dim=1)[:,i,:,:])
                precision_b1_seg.append(prec)
                recall_b1_seg.append(recall)
                acc_b1_seg.append(acc)
                iou_b1_seg.append(iou)

            running_iou_train_mean.append(np.mean(iou_seg))
            if idx%2 == 0 and gpu_id == 0:
                writer.add_scalar('Overall Loss/train', loss.item(), step_train)
                writer.add_scalar('Loss_cos/train', loss_cos.item(), step_train)
                writer.add_scalar('Loss_b1_seg/train', loss_b1_seg.item(), step_train)
                writer.add_scalar('Loss_seg/train', loss_seg.item(), step_train)
                writer.add_scalar('Mean IOU/train', np.mean(iou_seg), step_train)
                writer.add_scalar('Mean Precision/train', np.mean(precision_seg), step_train)
                writer.add_scalar('Mean Recall/train', np.mean(recall_seg), step_train)
                writer.add_scalar('Mean Acc/train', np.mean(acc_seg), step_train)
                    
                writer.add_scalar('Mean IOU B1 Seg/train', np.mean(iou_b1_seg), step_train)
                writer.add_scalar('Mean Precision B1 Seg/train', np.mean(precision_b1_seg), step_train)
                writer.add_scalar('Mean Recall B1 Seg/train', np.mean(recall_b1_seg), step_train)
                writer.add_scalar('Mean Acc B1 Seg/train', np.mean(acc_b1_seg), step_train)

                for i in range(len(class_names)):
                    writer.add_scalar(f'IOU_train/{i}', iou_seg[i], step_train)
                    writer.add_scalar(f'Precision_train/{i}', precision_seg[i], step_train)
                    writer.add_scalar(f'Recall_train/{i}', recall_seg[i], step_train)
                    writer.add_scalar(f'Acc_train/{i}', acc_seg[i], step_train)
                        
                    writer.add_scalar(f'IOU_B1_Seg_train/{i}', iou_b1_seg[i], step_train)
                    writer.add_scalar(f'Precision_B1_Seg_train/{i}', precision_b1_seg[i], step_train)
                    writer.add_scalar(f'Recall_B1_Seg_train/{i}', recall_b1_seg[i], step_train)
                    writer.add_scalar(f'Dice_B1_Seg_train/{i}', acc_b1_seg[i], step_train)


            

            step_train += 1

        print(f'IOU: {np.mean(running_iou_train_mean)}')

        if args['use_gpu']:
            model.module.save_model(exp, epoch)
        else:
            model.save_model(exp, epoch)

        model.eval()
        running_iou_mean = []
        for idx, (b1,b2,gt) in enumerate(tqdm(val_dataloader)):
            b1 = b1.to(device)
            b2 = b2.to(device)
            gt = gt.to(device)

            with torch.no_grad():
                b1_seg, b1_l, feature_l, seg = model(b1, b2)
                # what should the threshold be
                loss_cos = (1-cos(b1_l.flatten(-2), feature_l.flatten(-2))).mean()*args['cos_loss_weight']
                loss_b1_seg = loss_bce(b1_seg, seg['pred_masks'])*args['b1_seg_loss_weight']
                loss_seg = loss_bce(seg['pred_masks'], gt)*args['seg_loss_weight']
                loss = loss_b1_seg + loss_seg + Loss_cos
                precision_seg = []
                recall_seg = []
                iou_seg = []
                acc_seg = []

                for i in range(len(class_names)):
                    acc,recall,prec,iou = IOU(gt[:,i,:,:], seg['pred_masks'].softmax(dim=1)[:,i,:,:])
                    precision_seg.append(prec)
                    recall_seg.append(recall)
                    acc_seg.append(acc)
                    iou_seg.append(iou)

                precision_b1_seg = []
                recall_b1_seg = []
                iou_b1_seg = []
                acc_b1_seg = []

                for i in range(len(class_names)):
                    acc,recall,prec,iou = IOU(gt[:,i,:,:], b1_seg.softmax(dim=1)[:,i,:,:])
                    precision_b1_seg.append(prec)
                    recall_b1_seg.append(recall)
                    acc_b1_seg.append(acc)
                    iou_b1_seg.append(iou)

                
                if idx%2 == 0 and gpu_id == 0:
                    writer.add_scalar('Overall Loss/val', loss.item(), step_val)
                    writer.add_scalar('Loss_cos/val', loss_cos.item(), step_val)
                    writer.add_scalar('Loss_b1_seg/val', loss_b1_seg.item(), step_val)
                    writer.add_scalar('Loss_seg/val', loss_seg.item(), step_val)
                    writer.add_scalar('Mean IOU/val', np.mean(iou_seg), step_val)
                    writer.add_scalar('Mean Precision/val', np.mean(precision_seg), step_val)
                    writer.add_scalar('Mean Recall/val', np.mean(recall_seg), step_val)
                    writer.add_scalar('Mean Acc/val', np.mean(acc_seg), step_val)
                        
                    writer.add_scalar('Mean IOU B1 Seg/val', np.mean(iou_b1_seg), step_val)
                    writer.add_scalar('Mean Precision B1 Seg/val', np.mean(precision_b1_seg), step_val)
                    writer.add_scalar('Mean Recall B1 Seg/val', np.mean(recall_b1_seg), step_val)
                    writer.add_scalar('Mean Acc B1 Seg/val', np.mean(acc_b1_seg), step_val)

                    for i in range(len(class_names)):
                        writer.add_scalar(f'IOU_val/{i}', iou_seg[i], step_val)
                        writer.add_scalar(f'Precision_val/{i}', precision_seg[i], step_val)
                        writer.add_scalar(f'Recall_val/{i}', recall_seg[i], step_val)
                        writer.add_scalar(f'Acc_val/{i}', acc_seg[i], step_val)
                            
                        writer.add_scalar(f'IOU_B1_Seg_val/{i}', iou_b1_seg[i], step_val)
                        writer.add_scalar(f'Precision_B1_Seg_val/{i}', precision_b1_seg[i], step_val)
                        writer.add_scalar(f'Recall_B1_Seg_val/{i}', recall_b1_seg[i], step_val)
                        writer.add_scalar(f'Dice_B1_Seg_val/{i}', acc_b1_seg[i], step_val)

            

            
            step_val += 1

        print(f'IOU: {np.mean(running_iou_mean)}')
        
        if np.mean(running_iou_mean)>best_iou:
            best_iou = np.mean(running_iou_mean)
            if args['use_gpu']:
                model.module.save_model(exp, epoch, latest = False)
            else:
                model.save_model(exp, epoch, latest = False)




if __name__ == "__main__":
    config = yaml.safe_load(open('trial_config.yaml', 'r'))
    main(config)