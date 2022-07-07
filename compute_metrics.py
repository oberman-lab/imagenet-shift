import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import argparse

import torchvision.transforms as transforms

from metrics import get_metrics

transform_imagenet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
def load_ds_info(dataset, imagenet_folder):
    if dataset == 'imagenet':
        ds_info = {
            'name': 'imagenet',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': imagenet_folder
        }
    elif dataset == 'imagenet-cartoon':
        ds_info = {
            'name': 'imagenet-cartoon',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join('datasets', 'imagenet-cartoon')
        }
    elif dataset == 'imagenet-drawing':
        ds_info = {
            'name': 'imagenet-drawing',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join('datasets', 'imagenet-drawing')
        }
    elif dataset == 'imagenet-drawing-II':
        ds_info = {
            'name': 'imagenet-drawing-II',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join('datasets', 'imagenet-drawing-II')
        }
    elif dataset == 'imagenet-drawing-III':
        ds_info = {
            'name': 'imagenet-drawing-III',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join('datasets', 'imagenet-drawing-III')
        }
    elif dataset == 'imagenet-drawing-IV':
        ds_info = {
            'name': 'imagenet-drawing-IV',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join('datasets', 'imagenet-drawing-IV')
        }
    return ds_info

def get_loader(ds_info, shuffle=False):
    dataset_dir = ds_info['root_folder_datasets']
    dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=ds_info['transform'])
    dataset.samples = np.array(dataset.samples)
    dataset.targets = np.array(dataset.targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds_info['batch_size'], shuffle=shuffle, num_workers=4)
    return loader

def get_logits_labels(net, ds_info, device):
    dataloader = get_loader(ds_info)
    net.eval()
    net.to(device);
    acc = 0.0
    total = 0
    logits = []
    labels = torch.tensor([], dtype=int)
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Gathering Logits and Labels", leave=False):
            images, targets = data
            # compute features and logits
            logits_temp = net(images.to(device)).cpu()
            total += len(images)
            targets = [int(x) for x in targets]
            targets = torch.from_numpy(np.asarray(targets))
            logits.append(logits_temp.cpu())
        logits = torch.cat(logits, dim=0)
        labels = torch.from_numpy(dataloader.dataset.targets).long()
    return logits, labels


parser = argparse.ArgumentParser(description='Compute accuracy and ECE')
parser.add_argument('--imagenet_folder', type=str, default='path/to/imagenet/val', help="Path to ImageNet val folder")
args = parser.parse_args()

architectures = ['alexnet', 'vgg19_bn', 'resnet50', 'densenet121', 'resnext101_32x8d', 'wide_resnet50_2', 'vit_b_16', 'convnext_small']
datasets = ['imagenet', 'imagenet-cartoon', 'imagenet-drawing', 'imagenet-drawing-II', 'imagenet-drawing-III', 'imagenet-drawing-IV']

device = torch.device('cuda:0')
os.makedirs('results', exist_ok=True)
for architecture in tqdm(architectures, desc='Architectures'):
    net = torchvision.models.__dict__[architecture](pretrained=True)
    for dataset in tqdm(datasets, desc='Datasets', leave=False):
        ds_info = load_ds_info(dataset, args.imagenet_folder)
        if not os.path.exists(f'results/{architecture}_{dataset}.npy'):
            logits, labels = get_logits_labels(net, ds_info, device)
            softmaxes = torch.nn.Softmax(dim=1)(logits)
            results = get_metrics(softmaxes, labels)
            np.save(f'results/{architecture}_{dataset}.npy', results)