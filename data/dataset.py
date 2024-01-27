import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
# cityscapes_train = torchvision.datasets.Cityscapes(
#                             root = 'cityscapes_dataset',
#                             split='train', 
#                             mode = 'fine', 
#                             target_type = 'semantic',
#                             transform=transforms_cityscape, 
#                             target_transform=target_transforms_cityscape
#                         )

# cityscapes_val = torchvision.datasets.Cityscapes(
#                             root = 'cityscapes_dataset',
#                             split='val', 
#                             mode = 'fine', 
#                             target_type = 'semantic',
#                             transform=transforms_cityscape, 
#                             target_transform=target_transforms_cityscape
#                         )
# cityscapes_test = torchvision.datasets.Cityscapes(
#                             root = 'cityscapes_dataset',
#                             split='test', 
#                             mode = 'fine', 
#                             target_type = 'semantic',
#                             transform=transforms_cityscape, 
#                             target_transform=target_transforms_cityscape
#                         )

# 19 classes - check the distributition
# void - number is more (data imbalance)
# for faster results train on half 
# find unique sets to get the target
# read the cityscan paper to look for certsain points
# try to find the mask values

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transforms(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    target_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform, target_transforms

class dataset(Dataset):
    def __init__(
        self,
        csv_file, 
        image_size, 
        color_mapping,
        kaggle,
    ):
        csv = pd.read_csv(csv_file)
        self.kaggle = kaggle
        self.images = csv['image'].values
        self.foggy_images = csv['foggy_image'].values
        self.gts = csv['target'].values
        self.transforms, self.target_transforms = get_transforms(image_size)
        self.color_mapping = color_mapping
        self.image_size = image_size

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if not self.kaggle:
            image = self.transforms(Image.open(self.images[idx].replace('\\','/')))
            foggy_image = self.transforms(Image.open(self.foggy_images[idx].replace('\\','/')))
            gt = self.target_transforms(Image.open(self.gts[idx].replace('\\','/')))[:3]
        else:
            image = self.transforms(Image.open('/kaggle/input/cityscapes-dataset/'+self.images[idx].replace('\\','/')))
            foggy_image = self.transforms(Image.open('/kaggle/input/cityscapes-dataset/'+self.foggy_images[idx].replace('\\','/')))
            gt = self.target_transforms(Image.open('/kaggle/input/cityscapes-dataset/'+self.gts[idx].replace('\\','/')))[:3]

        gt_255 = (gt*255).to(torch.int)
        x = torch.zeros(26,self.image_size, self.image_size)
        for i in range(gt.shape[1]):
            for j in range(gt.shape[1]):
                try:
                    color = tuple(gt_255[:,i,j].numpy())
                    id = self.color_mapping[color]
                    x[id,i,j] = 255
                except:
                    x[0,i,j] = 255

        del gt_255
        del gt


        return image, foggy_image, x/255
