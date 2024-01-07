import gdown
import tarfile
import os
import logging
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
from PIL import Image
import random
import torch
import zipfile
import requests

class MiniImageNetSubDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

class MiniImageNetDataSet(torch.utils.data.Dataset):
    def __init__(self, dataDir, classLimit = -1, image_limit_per_class = -1):
        self.dataHomeFolder = dataDir
        self.images = []
        self.labels = []
        self.classes_names = set()
        
        all_classes = os.listdir(dataDir)
        if classLimit == -1:
            selected_classes = all_classes
        else:
            selected_classes = random.sample(all_classes, classLimit)
        for class_index, class_name in enumerate(selected_classes):
            class_path = os.path.join(self.dataHomeFolder, class_name)
            all_images = [img for img in os.listdir(class_path) if img.endswith('.jpg')]
            if image_limit_per_class == -1:
                selected_images = all_images
            else:
                selected_images = random.sample(all_images, image_limit_per_class)
            for image_name in selected_images:
                image_path = os.path.join(class_path, image_name)
                self.images.append(image_path)
                self.labels.append(class_index)
                self.classes_names.add(class_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        return image, label

class MiniImageNetDownloader:
    def __init__(self, url, miniImageNetLocation):
        self.url = url
        self.miniImageNetLocation = miniImageNetLocation
        self.logger = self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def download(self):
        if os.path.exists(self.miniImageNetLocation):
            self.logger.info('MiniImageNet database already exists.')
            return
        with TemporaryDirectory() as tempdir:
            tempTar = 'train.tar'
            gdown.download(self.url, tempTar, quiet=False)
            self.logger.info('Downloaded train.tar')
            if not os.path.exists(self.miniImageNetLocation):
                with tarfile.open(tempTar) as tar:
                    tar.extractall(path=self.miniImageNetLocation)
                    self.logger.info('Extracted train.tar to {}'.format(self.miniImageNetLocation))
                    

class EuroSatDownloader:
    def __init__(self, url, path):
        self.url = url
        self.path = path
        self.logger = self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def download(self):
        eurosat_rgb_directory = os.path.join(self.path, 'EuroSat_RGB')
        if os.path.exists(eurosat_rgb_directory):
            self.logger.info('EuroSat database already exists.')
            return
        with TemporaryDirectory() as tempdir:
            tempZip = 'EuroSAT_RGB.zip'
            response = requests.get(self.url)

            if response.status_code == 200:
                with open(tempZip, "wb") as file:
                    file.write(response.content)

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            with zipfile.ZipFile(tempZip, "r") as zip_ref:
                zip_ref.extractall(self.path)
        return eurosat_rgb_directory

                    
def createMiniImageNetDataLoaders(transforms, dataDir, split=0.8, batch_size=32):
    full_one_set = MiniImageNetDataSet(dataDir)
    
    # Calculate the sizes for train, validation, and test sets
    total_size = len(full_one_set)
    train_size = int(split * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_one_set, [train_size, val_size, test_size])
    
    train_dataset_transformed = MiniImageNetSubDataset(
        train_dataset, transform = transforms['train']
    )
    val_dataset_transformed = MiniImageNetSubDataset(
        val_dataset, transform = transforms['val']
    )
    test_dataset_transformed = MiniImageNetSubDataset(
        test_dataset, transform = transforms['test']
    )

    # Create data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset_transformed, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset_transformed, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset_transformed, batch_size=batch_size, shuffle=True)
    
    dataset_sizes = {
        'train': train_size,
        'test': test_size,
        'val': val_size
    }
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return dataloaders, full_one_set.classes_names, dataset_sizes


def createEuroSatDataLoaders(transforms, dataDir, split=0.25, batch_size=2, classLimit = 5, image_limit_per_class = 20):
    full_one_set = MiniImageNetDataSet(dataDir, classLimit = classLimit, image_limit_per_class = image_limit_per_class)
    
    # Calculate the sizes for train, validation, and test sets
    total_size = len(full_one_set)
    train_size = int(split * total_size)
    test_size = total_size - train_size

    # Split the dataset into train, validation, and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(full_one_set, [train_size, test_size])
    
    train_dataset_transformed = MiniImageNetSubDataset(
        train_dataset, transform = transforms['train']
    )
    test_dataset_transformed = MiniImageNetSubDataset(
        test_dataset, transform = transforms['test']
    )

    # Create data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset_transformed, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset_transformed, batch_size=batch_size, shuffle=False)
    
    dataset_sizes = {
        'train': train_size,
        'test': test_size,
    }
    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }
    return dataloaders, full_one_set.classes_names, dataset_sizes