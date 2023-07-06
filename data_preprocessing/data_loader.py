import logging

import numpy as np
from numpy.core.fromnumeric import mean
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data_preprocessing.datasets import CIFAR_truncated, ImageFolder_custom,\
ImageFolderTruncated, CifarReduced, ChestMINIST_truncated, ChestXray14, EuroSAT, PCAM_Reduced, PCAM_truncated
from PIL import Image

from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision
import torch.nn.functional as F

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class Lighting(object):
    imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
            ])
        }
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

def _data_transforms_cifar(datadir,img_size=32):
    if "cifar100" or "cifar-100"  in datadir:
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
    else:
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def _data_transforms_cinic10(datadir):
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
        lambda x: F.pad(x.unsqueeze(0),
        (4, 4, 4, 4),
        mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std),
    ])

    # Transformer for test set
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: F.pad(x.unsqueeze(0),
                            (4, 4, 4, 4),
                            mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,
                            std=cinic_std),
        ])
    return train_transform, valid_transform


def _data_transforms_imagenet(datadir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_scale = 0.08
    jitter_param = 0.4
    lighting_param = 0.1
    image_size = 224
    image_resize = 256

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
        transforms.ColorJitter(
            brightness=jitter_param, contrast=jitter_param,
            saturation=jitter_param),
        # Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, valid_transform

def _data_transforms_prompt(datadir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    crop_scale = 0.08
    jitter_param = 0.4
    lighting_param = 0.1
    image_size = 384
    image_resize = 400

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
        transforms.ColorJitter(
            brightness=jitter_param, contrast=jitter_param,
            saturation=jitter_param),
        # Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, valid_transform

def _data_transforms_chestminist(datadir, img_size=28):

    train_transform = valid_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomCrop(img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
    ])
   
    return train_transform, valid_transform

def _data_transforms_cheestxray(datadir, img_size=1024):
    
    train_mean = [0.4976, 0.4976, 0.4976]
    train_std = [0.2496, 0.2496, 0.2496]

    test_mean = [0.4715, 0.4715, 0.4715]
    test_std = [0.2406, 0.2406, 0.2406]

    train_transform = transforms.Compose([
        # transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
        transforms.Resize(img_size),
        transforms.RandomRotation(20), # Image augmentation
        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
        transforms.Normalize(train_mean, train_std),
    ])
    valid_transform = transforms.Compose([
        # transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
        transforms.Resize(img_size),
        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
        transforms.Normalize(test_mean, test_std),
    ])

    return train_transform, valid_transform

def _data_transforms_eurosat(datadir, img_size=1024):
    
    train_mean = [0.3448, 0.3806, 0.4081]
    train_std = [0.2014, 0.1351, 0.1135]

    test_mean = [0.3459, 0.3821, 0.4096]
    test_std = [0.2019, 0.1353, 0.1140]

    train_transform = transforms.Compose([
        # transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(), # Image augmentation
        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
        transforms.Normalize(train_mean, train_std),
    ])
    valid_transform = transforms.Compose([
        # transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
        transforms.Resize(img_size),
        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
        transforms.Normalize(test_mean, test_std),
    ])

    return train_transform, valid_transform

def _data_transforms_pcam(datadir, img_size=1024):
    
    train_mean = [0.1287, 0.0989, 0.1270]
    train_std = [0.2894, 0.2400, 0.2829]

    test_mean = [0.1248, 0.0957, 0.1273]
    test_std = [0.2827, 0.2354, 0.2830]

    train_transform = transforms.Compose([
        # transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(), # Image augmentation
        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
        transforms.Normalize(train_mean, train_std),
    ])
    valid_transform = transforms.Compose([
        # transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
        transforms.Resize(img_size),
        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
        transforms.Normalize(test_mean, test_std),
    ])

    return train_transform, valid_transform


def load_data(datadir, img_size=224, sample_num=-1):
    download = True
    if 'cifar' in datadir:
        if sample_num>-1:
            train_transform, test_transform = _data_transforms_cifar(datadir)
            dl_obj = CifarReduced
        else:
            train_transform, test_transform = _data_transforms_cifar(datadir)
            dl_obj = CIFAR_truncated
    elif 'cinic' in datadir:
        train_transform, test_transform = _data_transforms_cinic10(datadir)
        dl_obj = ImageFolderTruncated
    elif 'CropDisease' in datadir:
        train_transform, test_transform = _data_transforms_prompt(datadir)
        dl_obj = ImageFolder_custom
    elif 'chestmnist' in datadir:
        train_transform, test_transform = _data_transforms_chestminist(datadir)
        dl_obj = ChestMINIST_truncated
    elif 'chestxray' in datadir:
        train_transform, test_transform = _data_transforms_cheestxray(datadir, img_size)
        dl_obj = ChestXray14
    elif 'eurosat' in datadir:
        train_transform, test_transform = _data_transforms_eurosat(datadir, img_size)
        dl_obj = EuroSAT
    elif 'pcam' in datadir:
        train_transform, test_transform = _data_transforms_pcam(datadir, img_size)
        dl_obj = PCAM_truncated if sample_num == -1 else PCAM_Reduced
        download = False
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom

    if sample_num>-1:
        train_ds = dl_obj(datadir, sample_num,train=True, download=download, transform=train_transform)
        test_ds = dl_obj(datadir, sample_num, train=False, download=download, transform=test_transform)
    else:
        train_ds = dl_obj(datadir, train=True, download=True, transform=train_transform)
        test_ds = dl_obj(datadir, train=False, download=True, transform=test_transform)
    if 'pcam' in datadir and sample_num != -1:
        y_train, y_test = train_ds.target[train_ds.sample_idxs], test_ds.target
    else:
        y_train, y_test = train_ds.target, test_ds.target

    return (y_train, y_test)


def partition_data(datadir, partition, n_nets, alpha, sample_num=-1):
    logging.info("*********partition data***************")
    y_train, y_test = load_data(datadir, sample_num = sample_num)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    class_num = len(np.unique(y_train))

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = class_num
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                # while not proportions.all() > 0:
                #     proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return class_num, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(datadir, train_bs, test_bs, dataidxs=None, img_size=224, sample_num=-1):
    download = True
    if 'cifar' in datadir:
        
        train_transform, test_transform = _data_transforms_cifar(datadir,img_size)
        dl_obj = CIFAR_truncated if sample_num == -1 else CifarReduced
        workers=8
        persist=True
    elif 'cinic' in datadir:
        train_transform, test_transform = _data_transforms_cinic10(datadir)
        dl_obj = ImageFolderTruncated
        workers=0
        persist=False
    elif 'CropDisease' in datadir:
        train_transform, test_transform = _data_transforms_prompt(datadir)
        dl_obj = ImageFolder_custom
        workers=16
        persist=False
    elif 'chestmnist' in datadir:
        train_transform, test_transform = _data_transforms_chestminist(datadir, img_size)
        dl_obj = ChestMINIST_truncated
        workers=8
        persist=True
    elif 'chestxray' in datadir:
        train_transform, test_transform = _data_transforms_cheestxray(datadir, img_size)
        dl_obj = ChestXray14
        workers=8
        persist=True 
    elif 'eurosat' in datadir:
        train_transform, test_transform = _data_transforms_eurosat(datadir, img_size)
        dl_obj = EuroSAT
        workers=8
        persist=True 
    elif 'pcam' in datadir:
        train_transform, test_transform = _data_transforms_pcam(datadir, img_size)
        dl_obj = PCAM_truncated if sample_num == -1 else PCAM_Reduced
        workers=8
        persist=True 
        download=False
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom
        workers=8
        persist=True


    if sample_num>-1:
        train_ds = dl_obj(datadir, sample_num, dataidxs=dataidxs, train=True, transform=train_transform, download=download)
        test_ds = dl_obj(datadir, sample_num, train=False, transform=test_transform, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=workers, persistent_workers=persist)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=workers, persistent_workers=persist)
    else:
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=train_transform, download=download)
        test_ds = dl_obj(datadir, train=False, transform=test_transform, download=True)
    
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=workers, persistent_workers=persist)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=workers, persistent_workers=persist)

    return train_dl, test_dl

def get_dataloader_fast(datadir, train_bs, test_bs, base_train, base_test, dataidxs=None, img_size=224, sample_num=-1):

    if 'chestxray' in datadir:
        dl_obj_train = base_train.get_fast_dataset
        dl_obj_test = base_test
        workers=4
        persist=True 
    else:
        raise NotImplementedError("fast load is not supported.")


    if sample_num>-1:
        train_ds = dl_obj_train(sample_num, dataidxs=dataidxs)
        test_ds = base_test
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=workers, persistent_workers=persist)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=workers, persistent_workers=persist)
    else:
        train_ds = dl_obj_train(dataidxs=dataidxs)
        test_ds = base_test
    
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=workers, persistent_workers=persist)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=workers, persistent_workers=persist)

    return train_dl, test_dl

def load_partition_data(data_dir, partition_method, partition_alpha, client_number, batch_size, img_size=224, sample_num=-1):
    class_num, net_dataidx_map, traindata_cls_counts = partition_data(data_dir, partition_method, client_number, partition_alpha, sample_num=sample_num)

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(data_dir, batch_size, batch_size,img_size=img_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(train_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    if 'chestxray' in data_dir:
        train_transform, test_transform = _data_transforms_cheestxray(data_dir, img_size)

        base_train = ChestXray14(data_dir, dataidxs=None, train=True, transform=train_transform, download=True)
        base_test = ChestXray14(data_dir, train=False, transform=test_transform, download=True)

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        if 'chestxray' in data_dir:
            train_data_local, test_data_local = get_dataloader_fast(data_dir, batch_size, batch_size, base_train, base_test, dataidxs, img_size, sample_num=sample_num)
        else:
            train_data_local, test_data_local = get_dataloader(data_dir, batch_size, batch_size, dataidxs,img_size, sample_num=sample_num)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
