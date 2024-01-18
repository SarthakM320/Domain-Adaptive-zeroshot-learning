from .utils_own import transforms_cityscape
import torchvision

cityscapes_dataset_train = torchvision.datasets.Cityscapes(
                            root = '.',
                            split='train', 
                            mode = 'fine', 
                            target_type = 'semantic',
                            transforms=transforms_cityscape
                        )

cityscapes_dataset_val = torchvision.datasets.Cityscapes(
                            root = '.',
                            split='val', 
                            mode = 'fine', 
                            target_type = 'semantic',
                            transforms=transforms_cityscape
                        )
cityscapes_dataset_test = torchvision.datasets.Cityscapes(
                            root = '.',
                            split='test', 
                            mode = 'fine', 
                            target_type = 'semantic',
                            transforms=transforms_cityscape
                        )