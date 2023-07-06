import os
from datasets.video_transforms import *
from datasets.kinetics import VideoClsDataset, VideoMAE,DataAugmentationForVideoMAE
import logging
from datasets.videomae_transforms import GroupMultiScaleCrop, GroupNormalize, Stack, ToTorchFormatTensor
#from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=True,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    #print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(args,split):
    mode=split
    if args.DATASET == 'UCF_CRIME':
        anno_path = None
        if  split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode=False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode=True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 13
    elif args.DATASET == 'GYM':
        anno_path = None
        if  split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode=False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode=True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 99
    elif args.DATASET == 'TOYO':
        anno_path = None
        if split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode = False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode = True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 31
    elif args.DATASET == 'K400':
        anno_path = None
        if split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode = False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode = True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    elif args.DATASET == 'SSV2':
        anno_path = None
        if split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode = False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode = True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174
    elif args.DATASET == 'UCF101':
        anno_path = None
        if split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode = False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode = True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='./',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.DATASET == 'HRI':
        #mode = None
        anno_path = None
        if split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode = False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode = True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 30
    elif args.DATASET == 'COIN':
        #mode = None
        anno_path = None
        if split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode = False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 180



    elif args.DATASET == 'INHARD':
        #mode = None
        anno_path = None
        if split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode = False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode = True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 30
    elif args.DATASET == 'HMDB51':
        #mode = None
        anno_path = None
        if split == 'train':
            anno_path = os.path.join(args.data_path, 'train.txt')
            test_mode = False
        elif split == "validation":
            anno_path = os.path.join(args.data_path, 'val.txt')
            test_mode = True
        elif split == 'test':
            anno_path = os.path.join(args.data_path, 'test.txt')
            test_mode = True

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if mode == "train" else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def record_net_data_stats(y_train, net_dataidx_map):
    #print('coming')
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    #print('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def build_training_dataset(args, data_ids):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=os.path.join(args.data_path, 'train.txt'),
        video_ext=args.ext,
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        data_ids=data_ids
    )
    #print("Data Aug = %s" % str(transform))
    return dataset

def fed_data_split(train_dataset, test_dataset_loader,cfg,n_nets=4,alpha=0.25,partition='hetero'):
    #train_dataset = build_training_dataset(cfg, [])
    train_targets=  np.array(train_dataset._labels)
    #print(train_targets)
    n_train = len(train_targets)
    class_num = len(np.unique(train_targets))
    #print(n_train,class_num)
    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    elif partition == "hetero":
        min_size = 0
        K = class_num
        N = n_train
        print("N = " + str(N))
        net_dataidx_map = {}
        while min_size < 10:
            #print(n_nets)
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                #print(k, targets)
                idx_k = np.where(train_targets == k)[0]
                #print(idx_k)
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                #print(min_size)
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(train_targets, net_dataidx_map)
    print("Class Count: ", traindata_cls_counts)
    #exit()
    client_number=n_nets
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    print("Train_DATA: ",train_data_num)
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        #print(dataidxs)
        train_data_client= build_training_dataset(cfg, np.array(dataidxs))
        train_data_local=fed_data_loader(train_data_client,cfg)
        test_data_local= test_dataset_loader #construct_loader(cfg, split="validation")
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

   # return class_num, net_dataidx_map, traindata_cls_counts

def fed_data_loader(dataset, args):
    data_loader_train = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )
    return data_loader_train

def construct_loader(args, split):
    if split == "validation":
        dataset, _ = build_dataset(args,split)
    elif split == "test":
        dataset, _ = build_dataset(args,split)
    elif split =="train":
        dataset= build_training_dataset(args, data_ids=[])
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )
    return data_loader, dataset


