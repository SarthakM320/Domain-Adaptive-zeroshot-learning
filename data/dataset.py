from .utils_own import transforms_cityscape, target_transforms_cityscape
import torchvision
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

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

class dataset(Dataset):
    def __init__(
        self,
        csv_file, 
        kaggle,
        transforms = transforms_cityscape,
        target_transforms = target_transforms_cityscape
    ):
        csv = pd.read_csv(csv_file)
        self.kaggle = kaggle
        self.images = csv['image'].values
        self.foggy_images = csv['foggy_image'].values
        self.gts = csv['target'].values
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if not self.kaggle:
            image = self.transforms(Image.open(self.images[idx]))
            foggy_image = self.transforms(Image.open(self.foggy_images[idx]))
            gt = self.target_transforms(Image.open(self.gts[idx]))
        else:
            image = self.transforms('/kaggle/input/cityscapes-dataset/'+Image.open(self.images[idx]))
            foggy_image = self.transforms('/kaggle/input/cityscapes-dataset/'+Image.open(self.foggy_images[idx]))
            gt = self.target_transforms('/kaggle/input/cityscapes-dataset/'+Image.open(self.gts[idx]))

        return image, foggy_image, gt
