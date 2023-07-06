from easydict import EasyDict
import torch
import numpy as np

import random
import argparse

import wandb

from torch.multiprocessing import Pool, Process, set_start_method, Queue, Lock

from models_video.build import build_model
import logging
logging.basicConfig(level = logging.INFO)


import os
from collections import defaultdict
import time

from datasets.video_datasets import fed_data_split, construct_loader

import methods.adapter as adapter

import data_preprocessing.custom_multiprocess as cm
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision.models import shufflenet_v2_x0_5


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Method settings
    parser.add_argument('--method', type=str, default='prompt', metavar='M',
                        help='baseline method')
    # parser.add_argument('--prompt_num', type=int, default=10, metavar='N',
    #                     help='prompt number for vpt')
    parser.add_argument('--ffn_adapt', default=False, action='store_true', help='whether activate AdaptFormer')
    parser.add_argument('--st_adapt', default=False, action='store_true', help='whether activate STAdaptFormer')
    parser.add_argument('--scalar', default=1.0, type=float, help='scaling the adapter effect')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    parser.add_argument('--vpt_num', default=8, type=int, help='number of VPT prompts')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--inception', default=False, action='store_true', help='whether use INCPETION mean and std'
                                                                                '(for Jx provided IN-21K pretrain')
    parser.add_argument('--finetune', default='./videomae/pretrain_videomae_ft_k400_1600_new.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--pretrained', default=False,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    #####################Training Settings##################################
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--lr', type=float, default=0.006, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=8, metavar='EP', help='how many epochs will be trained locally')
    parser.add_argument('--comm_round', type=int, default=20, help='how many round of communications we shoud use')
    parser.add_argument('--optimizer', default='SGD',type=str,
                        help='selection of optimizer')

    #######Data Settings########
    parser.add_argument('--DATASET', type=str, default='UCF101')
    parser.add_argument('--nb_classes', default=101, type=int,
                        help='number of the classification types')
    parser.add_argument('--ext', type=str, default='mp4')
    parser.add_argument('--data_path', type=str, default='./',
                        help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
                        help='partition alpha (default: 0.5)')
    parser.add_argument('--client_number', type=int, default=32, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    ######################Custom Parameters#########################3
    parser.add_argument('--linprob', default=True)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--window_size', default=(4, 14, 14))
    parser.add_argument('--patch_size', type=int, default=(14, 14))
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='No drop path for linear probe')
    parser.add_argument('--use_mean_pooling', default=True)
    parser.add_argument('--init_scale', default=0.001, type=float)

    # video data parameters
    parser.add_argument('--data_set', default='UCF101',
                        choices=['SSV2', 'HMDB51', 'image_folder'],
                        type=str, help='dataset')  ## Changeable
    parser.add_argument('--data_fraction', type=float, default=1.0)
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_sample', type=int, default=1,
                        help='Repeated_aug (default: 1)')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=4)
    parser.add_argument('--test_num_crop', type=int, default=3)
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')

    ###Federated Settings

    parser.add_argument('--mu', type=float, default=0.001, metavar='MU',
                        help='mu (default: 0.001)')
    parser.add_argument('--rho', type=float, default=0.05, metavar='PHO',
                        help='pho for SAM (default: 0.05)')
    parser.add_argument('--width', type=float, default=0.7, metavar='WI',
                        help='minimum width for mutual training')
    parser.add_argument('--mult', type=float, default=1.0, metavar='MT',
                        help='multiplier for mutual training')
    parser.add_argument('--num_subnets', type=int, default=3,
                        help='how many subnets for mutual training')
    parser.add_argument('--localbn', action='store_true', default=False,
                        help='Keep local BNs each round')
    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')
    parser.add_argument('--thread_number', type=int, default=4, metavar='NN',
                        help='number of threads in a distributed cluster')
    parser.add_argument('--client_sample', type=float, default=1, metavar='MT',
                        help='Fraction of clients to sample')
    parser.add_argument('--resolution_type', type=int, default=0,
                        help='Specifies the resolution list used')
    parser.add_argument('--stoch_depth', default=0.5, type=float,
                        help='stochastic depth probability')
    parser.add_argument('--beta', default=0.0, type=float,
                        help='hyperparameter beta for mixup')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='reduce the sampler number for debugging')

    parser.add_argument('--stat', action='store_true', default=False,
                        help='show the state of model')
    parser.add_argument('--dp', default=False,action='store_true', help='Apply differential privacy')
    parser.add_argument('--delta', default=1e-3, type=float,
                        help='hyperparameter delta for DP')
    parser.add_argument('--epsilon', default=5, type=float,
                        help='hyperparameter epsilon for DP')
    parser.add_argument('--max_grad_norm', default=1, type=float,
                        help='hyperparameter grad_norm for DP')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='save the server model')
    parser.add_argument('--sam_mode', type=str, default='none', choices=['asam', 'sam', 'none'], metavar='N',
                        help='type of sam')
    parser.add_argument('--ffn_option', type=str, default='parallel')
    parser.add_argument('--sample_num', type=int, default=-1, metavar='N',
                        help='how many sample will be trained in total. -1 for no reduce')

    args = parser.parse_args()

    return args


# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# Helper Functions
def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])
    # client.server =


def run_clients(received_info):
    return client.run(received_info)


def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample < 1.0:
            num_clients = int(args.client_number * args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients<args.thread_number:
            args.thread_number=num_clients
        if num_clients % args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients / args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t + clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            logging.info(
                "############ WARNING: Sampled client number not divisible by number of threads ##############")
            break
    return mapping_dict


def datapath2str(path):
    if "cifar-100" in path or "cifar100" in path:
        return 'Cifar100'
    elif "CropDisease" in path:
        return 'CropDisease'


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args.ffn_adapt,
        st_adapt=args.st_adapt,
        ffn_option=args.ffn_option,
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar=str(args.scalar),
        ffn_num=args.ffn_num,
        d_model=768,
        # VPT related
        vpt_on=args.vpt,
        vpt_num=args.vpt_num,
    )

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed()
    # get arguments

    # data_name = datapath2str(args.DATASET)
    save_path = './logs/{}_{}_{}_lr{:.0e}_e{}_c{}_s{}_d{}_p{}_a{}_{}_{}_{}_{}'.format(args.ffn_num,args.vpt_num,args.method, args.lr, args.epochs,
                                                                          args.client_number, args.client_sample,
                                                                          args.DATASET, args.scalar,
                                                                          args.partition_alpha,
                                                                          args.nb_classes,args.optimizer,args.finetune,
                                                                          time.strftime("%Y-%m-%d_%H-%M-%S",
                                                                                        time.localtime()))

    # save_path = './logs/{}_lr{:.0e}_e{}_c{}_{}_{}'.format(args.method, args.lr, args.epochs, args.client_number,
    #                                                       args.nb_classes,
    #                                                       time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    args.save_path = save_path
    if not args.debug:
        wandb.init(config=args,project='sampling')
        wandb.run.name = save_path #.split(os.path.sep)[-1]

    train_data_global, train_dataset = construct_loader(args, "train")
    test_data_global, test_dataset = construct_loader(args, "validation")
    train_data_num, data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = fed_data_split(
        train_dataset, test_data_global, args, args.client_number, args.partition_alpha, args.partition_method)
    mapping_dict = allocate_clients_to_threads(args)
    # NOTE Always use fedavg right now
    if args.method == 'scratch':
        if args.fulltune:
            print("Training the complete model")
        Server = adapter.Server
        Client = adapter.Client
        # basic_model = (args.vit_type, num_classes= class_num, pretrained= True)
        Model = build_model
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num, 'tuning_config': tuning_config}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                        'tuning_config': tuning_config} for i in range(args.thread_number)]
    elif args.method == 'bias':
        print("Bias+ Linear FineTuning")
        Server = adapter.Server
        Client = adapter.Client
        # basic_model = (args.vit_type, num_classes= class_num, pretrained= True)
        Model = build_model
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num, 'tuning_config': tuning_config}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                        'tuning_config': tuning_config} for i in range(args.thread_number)]

    elif args.method == 'prompt':
        if args.vpt:
            print("Prompt-Model")
            Server = adapter.Server
            Client = adapter.Client
            # basic_model = (args.vit_type, num_classes= class_num, pretrained= True)
            Model = build_model
            server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                           'num_classes': class_num, 'tuning_config': tuning_config}
            client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                            'device': i % torch.cuda.device_count(),
                            'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                            'tuning_config': tuning_config} for i in range(args.thread_number)]
    elif args.method == 'linear_head':
        if args.vpt == False and args.ffn_adapt == False and args.st_adapt == False:
            print("Fine-Tuning Penultimate Layer")
            Server = adapter.Server
            Client = adapter.Client
            # basic_model = (args.vit_type, num_classes= class_num, pretrained= True)
            Model = build_model
            server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                           'num_classes': class_num, 'tuning_config': tuning_config}
            client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                            'device': i % torch.cuda.device_count(),
                            'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                            'tuning_config': tuning_config} for i in range(args.thread_number)]

    elif args.method == 'st_adapter':
        if args.ffn_adapt:
            print("AdaptFormer Finetuning")
        elif args.st_adapt:
            print("ST_adapter Finetuning")
        Server = adapter.Server
        Client = adapter.Client
        # basic_model = (args.vit_type, num_classes= class_num, pretrained= True)
        Model = build_model
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num, 'tuning_config': tuning_config}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                        'tuning_config': tuning_config} for i in range(args.thread_number)]
    elif args.method == 'ffn_adapter':
        print("AdaptFormer Finetuning")
        Server = adapter.Server
        Client = adapter.Client
        # basic_model = (args.vit_type, num_classes= class_num, pretrained= True)
        Model = build_model
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num, 'tuning_config': tuning_config}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                        'tuning_config': tuning_config} for i in range(args.thread_number)]



    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')

    if args.pretrained:
        print('Pretrained')
        server_dict['save_path'] = "best.pt"
        server = Server(server_dict, args)
        server.model.load_state_dict(torch.load('logs/2022-03-21 20:56:10.626330_aavg_e20_c16/server.pt'))
        acc = server.test()
    else:
        # init server
        server_dict['save_path'] = save_path
        if not os.path.exists(server_dict['save_path']):
            os.makedirs(server_dict['save_path'])
        server = Server(server_dict, args)
        server_outputs = server.start()

        param_num = get_parameter_number(server.model)
        if not args.debug:
            wandb.log({"Params/Total": param_num["Total"], "Params/Trainable":param_num["Trainable"]})
        # Start Federated Training
        # init nodes
        client_info = Queue(32)
        # clients = {}
        for i in range(args.thread_number):
            client_info.put((client_dict[i], args))
            # clients[i] = Client(client_dict[i], args)

        # Start server and get initial outputs
        pool = cm.MyPool(args.thread_number, init_process, (client_info, Client))
        logging.info("I am going to start")

        # if args.debug:
        #     time.sleep(10 * (args.client_number * args.client_sample / args.thread_number))
        # else:
        #     time.sleep(150 * (args.client_number * args.client_sample / args.thread_number)) #  Allow time for threads to start up
        for r in range(args.comm_round):
            #logging.info('************** Round: {} ***************'.format(r))
            print('************** Round: {} ***************'.format(r), flush=True)
            round_start = time.time()
            client_outputs = pool.map(run_clients, server_outputs)
            client_outputs = [c for sublist in client_outputs for c in sublist]
            server_outputs = server.run(client_outputs)
            round_end = time.time()
            # if not args.debug:
            #     wandb.log({"Round Time": round_end-round_start, "round": r})
            out_str = ' Round {} Time: {}s \n'.format(r, round_end - round_start)
            #logging.info(out_str)
            #print(out_str)
            with open('{}/out.log'.format(args.save_path), 'a+') as out_file:
                out_file.write(out_str)
            # time.sleep(10)
        pool.close()
        pool.join()
