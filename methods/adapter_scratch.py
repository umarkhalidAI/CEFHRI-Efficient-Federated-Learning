import torch
# import wandb
import logging
from methods.base_scratch import Base_Client, Base_Server
from torch.multiprocessing import current_process
from data_preprocessing.sam import SAM
# from data_preprocessing.dpsgd import DPSGD
from collections import OrderedDict
import os
from timm.models.layers import trunc_normal_
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed


# from methods.natural_gradient_optimizer import NGF_Optimizer

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(args,
                                     pretrained=False,
                                     num_classes=args.nb_classes,
                                     all_frames=args.num_frames * args.num_segments,
                                     tubelet_size=args.tubelet_size,
                                     drop_rate=args.drop,
                                     drop_path_rate=args.drop_path,
                                     attn_drop_rate=args.attn_drop_rate,
                                     # drop_block_rate=None,
                                     use_mean_pooling=args.use_mean_pooling,
                                     init_scale=args.init_scale,
                                     tuning_config=self.tuning_config, batch=args.batch_size)  # .to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss_scaler = NativeScaler()
        params = self._pre_train_weights(args)  # filter(lambda p: p.requires_grad,self.model.parameters())
        if args.optimizer == 'SGD':
            base_optimizer = torch.optim.SGD
            kwargs = dict(lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay, nesterov=True)
            # self.optimizer = base_optimizer(params, **kwargs)
        elif args.optimizer == 'adamw':
            base_optimizer = torch.optim.AdamW
            kwargs = dict(lr=self.args.lr, weight_decay=self.args.weight_decay)
        if args.optimizer != "NGF":
            if args.sam_mode == 'none':
                self.optimizer = base_optimizer(params, **kwargs)
            elif args.sam_mode == 'asam':
                self.optimizer = SAM(params, base_optimizer, rho=self.args.rho, adaptive=True, **kwargs)
            elif args.sam_mode == 'sam':
                self.optimizer = SAM(params, base_optimizer, rho=self.args.rho, adaptive=False, **kwargs)

    def _pre_train_weights(self, args):
        if args.finetune:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            # print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'model' in checkpoint:
                raw_checkpoint_model = checkpoint['model']
            elif 'module' in checkpoint:
                raw_checkpoint_model = checkpoint['module']
            else:
                raw_checkpoint_model = checkpoint
            # TODO: refine
            if os.path.basename(args.finetune).startswith('pretrain'):
                checkpoint_model = OrderedDict()
                for k, v in raw_checkpoint_model.items():
                    if k.startswith('encoder.'):
                        checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
                    elif k.startswith('fc_norm.'):
                        checkpoint_model[k[3:]] = v
                    else:
                        checkpoint_model[k] = v
                    if k == 'head.weight':
                        del checkpoint_model['head.weight']
                    if k == 'head.bias':
                        del checkpoint_model['head.bias']
                del checkpoint_model['norm.weight']
                del checkpoint_model['norm.bias']
            elif os.path.basename(args.finetune).startswith('finetune'):
                checkpoint_model = raw_checkpoint_model
            elif os.path.basename(args.finetune) == "vit_base_patch16_224_in21k_tongzhan_new.pth":
                checkpoint_model = raw_checkpoint_model
                del checkpoint_model['norm.weight']
                del checkpoint_model['norm.bias']
            elif os.path.basename(args.finetune).startswith('swin_base_patch244'):
                checkpoint_model = OrderedDict()
                for k, v in raw_checkpoint_model['state_dict'].items():
                    if k.startswith('backbone.'):
                        checkpoint_model[k[9:]] = v
            else:
                raise ValueError("Warning: Double Check!")

            state_dict = self.model.state_dict()
            interpolate_pos_embed(self.model, checkpoint_model)
            # load pre-trained model
            msg = self.model.load_state_dict(checkpoint_model, strict=False)
            # manually initialize fc layer: following MoCo v3
            trunc_normal_(self.model.head.weight, std=0.01)

        # hack: revise model's head with BN
        self.model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(self.model.head.in_features, affine=False, eps=1e-6),
                                              self.model.head)
        if not args.resume:
            if args.method != "bias" and args.finetune != "":
                # freeze all but the head
                for name, p in self.model.named_parameters():
                    if name in msg.missing_keys:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False if not args.fulltune else True
                for _, p in self.model.head.named_parameters():
                    p.requires_grad = True
                for _, p in self.model.fc_norm.named_parameters():
                    p.requires_grad = True
            elif args.method == "scratch" and args.finetune == "":
                for name, p in self.model.named_parameters():
                    p.requires_grad = True
                for _, p in self.model.head.named_parameters():
                    p.requires_grad = True
                for _, p in self.model.fc_norm.named_parameters():
                    p.requires_grad = True

            if args.method == "bias":
                for k, p in self.model.named_parameters():
                    if 'bias' not in k:
                        p.requires_grad = False
                for p in self.model.head.parameters():
                    p.requires_grad = True
        bn_params = []
        # Non-batchnorm parameters.
        non_bn_parameters = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if "bn" in name:
                    bn_params.append(p)
                else:
                    non_bn_parameters.append(p)
        optim_params = [
            {"params": bn_params, "weight_decay": 0.},
            {"params": non_bn_parameters, "weight_decay": args.weight_decay},
        ]
        return optim_params


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(args,
                                     pretrained=False,
                                     num_classes=args.nb_classes,
                                     all_frames=args.num_frames * args.num_segments,
                                     tubelet_size=args.tubelet_size,
                                     drop_rate=args.drop,
                                     drop_path_rate=args.drop_path,
                                     attn_drop_rate=args.attn_drop_rate,
                                     # drop_block_rate=None,
                                     use_mean_pooling=args.use_mean_pooling,
                                     init_scale=args.init_scale,
                                     tuning_config=self.tuning_config, batch=args.batch_size).to(self.device)
        # if not self.args.debug:
        #     wandb.watch(self.model)
        self._state_dict(args)

    def _state_dict(self, args):
        if args.finetune:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            # print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'model' in checkpoint:
                raw_checkpoint_model = checkpoint['model']
            elif 'module' in checkpoint:
                raw_checkpoint_model = checkpoint['module']
            else:
                raw_checkpoint_model = checkpoint
            # TODO: refine
            if os.path.basename(args.finetune).startswith('pretrain'):
                checkpoint_model = OrderedDict()
                for k, v in raw_checkpoint_model.items():
                    if k.startswith('encoder.'):
                        checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
                    elif k.startswith('fc_norm.'):
                        checkpoint_model[k[3:]] = v
                    else:
                        checkpoint_model[k] = v
                    if k == 'head.weight':
                        del checkpoint_model['head.weight']
                    if k == 'head.bias':
                        del checkpoint_model['head.bias']
                del checkpoint_model['norm.weight']
                del checkpoint_model['norm.bias']
            elif os.path.basename(args.finetune).startswith('finetune'):
                checkpoint_model = raw_checkpoint_model
            elif os.path.basename(args.finetune) == "vit_base_patch16_224_in21k_tongzhan_new.pth":
                checkpoint_model = raw_checkpoint_model
                del checkpoint_model['norm.weight']
                del checkpoint_model['norm.bias']
            elif os.path.basename(args.finetune).startswith('swin_base_patch244'):
                checkpoint_model = OrderedDict()
                for k, v in raw_checkpoint_model['state_dict'].items():
                    if k.startswith('backbone.'):
                        checkpoint_model[k[9:]] = v
            else:
                raise ValueError("Warning: Double Check!")

            state_dict = self.model.state_dict()
            interpolate_pos_embed(self.model, checkpoint_model)
            # load pre-trained model
            msg = self.model.load_state_dict(checkpoint_model, strict=False)
            # manually initialize fc layer: following MoCo v3

            trunc_normal_(self.model.head.weight, std=0.01)

        # hack: revise model's head with BN
        self.model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(self.model.head.in_features, affine=False, eps=1e-6),
                                              self.model.head)
        if not args.resume:
            if args.method != "bias" and args.finetune != "":
                # freeze all but the head
                for name, p in self.model.named_parameters():
                    if name in msg.missing_keys:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False if not args.fulltune else True
                for _, p in self.model.head.named_parameters():
                    p.requires_grad = True
                for _, p in self.model.fc_norm.named_parameters():
                    p.requires_grad = True
            elif args.method == "scratch" and args.finetune == "":
                for name, p in self.model.named_parameters():
                    p.requires_grad = True
                for _, p in self.model.head.named_parameters():
                    p.requires_grad = True
                for _, p in self.model.fc_norm.named_parameters():
                    p.requires_grad = True
            if args.method == "bias":
                for k, p in self.model.named_parameters():
                    if 'bias' not in k:
                        p.requires_grad = False
                for p in self.model.head.parameters():
                    p.requires_grad = True
        # n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # pytorch_total_params = sum(p.numel() for p in self.model.parameters())

        # print("Model = %s" % str(model_without_ddp))
        # print("Com-Cost: %.5f" %(n_parameters*4*32/1073741824))
        # print('number of params (M): %.2f' % (n_parameters/ 1.e6))
        # print('Total number of params (M): %.2f' % (pytorch_total_params / 1.e6))
        # exit()


if __name__ == '__main__':
    from main import add_args, allocate_clients_to_threads
    from methods import prompt
    from models.vpt_official import build_promptmodel
    import timm
    import argparse
    import data_preprocessing.data_loader as dl

    parser = argparse.ArgumentParser()
    args = add_args(parser)

    img_size = 224 if '224' in args.vit_type else 384
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number,
                                       args.batch_size, img_size)

    mapping_dict = allocate_clients_to_threads(args)

    Server = prompt.Server
    Client = prompt.Client
    basic_model = timm.create_model(args.vit_type, num_classes=class_num, pretrained=True)
    Model = build_promptmodel

    server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                   'num_classes': class_num, 'basic_model': basic_model}
    client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                    'device': i % torch.cuda.device_count(),
                    'client_map': mapping_dict[i], 'model_type': Model, 'basic_model': basic_model,
                    'num_classes': class_num} for i in range(args.thread_number)]

    client = Client(client_dict[0], args)
    client.model.obtain_prompt()