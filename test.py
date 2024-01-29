import torch 
from torch import nn
from data import test_dataset, labels
import numpy as np
import random
from models import CLIPVisionTransformer, CLIPResNet, CLIPTextEncoder, Decoder, SegDecoder, OverallModel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm 
import json 
import warnings
import yaml
from metrics import IoU


warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    if args['use_gpu']:
        gpu_id = 0
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

    test_data = test_dataset(
        'dataset/cityscape_test.csv', 
        image_size = args['img_size'],
        num_classes = args['num_classes'],
        kaggle = args['kaggle']
    )

    test_dataloader = DataLoader(test_data, batch_size=args['batch_size'], num_workers = 4,shuffle=False)
   
    exp=f'{args["folder"]}/'+args['exp_name']
    
    writer = SummaryWriter(exp)
    with open(f'{exp}/params.json', 'w') as f:
        json.dump(args, f)

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

    model.load_model('Experiments/trial_2', latest = False)

    loss_bce = nn.CrossEntropyLoss()
    cos = nn.CosineSimilarity(dim = -1)

    model.to(device)
    model.eval()

    optim = torch.optim.Adam(model.parameters(), lr = args['lr'])
    
    step_test = 0
    best_iou = 0
    running_iou_mean = []
    iou_per_class = np.zeros(args['num_classes'])

    stats = {
        'mean_iou':[],
        'mean_precision':[],
        'mean_recall':[],
        'mean_acc':[],
        'iou':[],
        'precision':[],
        'recall':[],
        'acc':[],
        'Overall IOU':0
    }

    with torch.no_grad():

        for idx, (b2, gt, mask) in enumerate(tqdm(test_dataloader)):
            b2 = b2.to(device)
            gt = gt.type(torch.LongTensor).to(device)
            b2_seg = model.forward_test(b2)

            loss = loss_bce(b2_seg, gt)
            precision_seg = []
            recall_seg = []
            iou_seg = []
            acc_seg = []

            seg = b2_seg.softmax(dim=1).cpu()

            for i in range(len(class_names)):
                acc,recall,prec,iou = IOU(mask[:,i,:,:], seg[:,i,:,:])
                precision_seg.append(prec.item())
                recall_seg.append(recall.item())
                acc_seg.append(acc.item())
                iou_seg.append(iou.item())

            running_iou_mean.append(np.mean(iou_seg))
            
            stats['mean_iou'].append(np.mean(iou_seg).astype(np.float64))
            stats['mean_precision'].append(np.mean(precision_seg).astype(np.float64))
            stats['mean_recall'].append(np.mean(recall_seg).astype(np.float64))
            stats['mean_acc'].append(np.mean(acc_seg).astype(np.float64))
            stats['iou'].append(iou_seg)
            stats['precision'].append(precision_seg)
            stats['acc'].append(acc_seg)
            stats['recall'].append(recall_seg)


            writer.add_scalar('Overall Loss', loss.item(), step_test)
            writer.add_scalar('Mean IOU', np.mean(iou_seg), step_test)
            writer.add_scalar('Mean Precision', np.mean(precision_seg), step_test)
            writer.add_scalar('Mean Recall', np.mean(recall_seg), step_test)
            writer.add_scalar('Mean Acc', np.mean(acc_seg), step_test)


            if idx == 0:
                iou_per_class += np.array(iou_seg)
            else:
                iou_per_class += (np.array(iou_seg)*idx)/(idx+1)
            for i in range(len(class_names)):
                
                writer.add_scalar(f'IOU/{i}', iou_seg[i], step_test)
                writer.add_scalar(f'Precision/{i}', precision_seg[i], step_test)
                writer.add_scalar(f'Recall/{i}', recall_seg[i], step_test)
                writer.add_scalar(f'Acc/{i}', acc_seg[i], step_test)

            step_test += 1
        
        print(f'IOU: {np.mean(running_iou_mean)}')
        stats['Overall IOU'] = np.mean(running_iou_mean).astype(np.float64)
        stats['Mean IOU per class'] = list(iou_per_class.astype(np.float64))
        print(stats)
        with open(f'{exp}/stats.json', 'w') as f:
            json.dump(stats, f)

if __name__ == "__main__":
    config = yaml.safe_load(open('trial_config.yaml', 'r'))
    config['exp_name'] += '_test'
    main(config)

