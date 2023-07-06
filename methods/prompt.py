import torch
import wandb
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
from data_preprocessing.sam import SAM



class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        img_size = 224 if '224' in args.vit_type else 384
        patch_size = 16 if '16' in args.vit_type else 32
        self.model = self.model_type(basic_model=client_dict['basic_model'], num_classes=self.num_classes,VPT_type=args.vpt_type,Prompt_Token_num=args.prompt_num,edge_size=img_size,patch_size=patch_size).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)


        params = filter(lambda p: p.requires_grad,self.model.parameters())
        if args.optimizer == 'sgd':

                base_optimizer = torch.optim.SGD
                kwargs = dict(lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

        elif args.optimizer == 'adamw':
            base_optimizer = torch.optim.AdamW
            kwargs = dict(lr=self.args.lr, weight_decay=self.args.wd)

        
        if args.sam_mode == 'none':
            self.optimizer = base_optimizer(params, **kwargs)
        elif args.sam_mode == 'asam':
            self.optimizer = SAM(params, base_optimizer, rho=self.args.rho, adaptive=True, **kwargs)
        elif args.sam_mode == 'sam':
            self.optimizer = SAM(params, base_optimizer, rho=self.args.rho, adaptive=False, **kwargs)
        



    def load_client_state_dict(self, server_state_dict):
        if self.args.localbn:
            server_dict = {k: v for k, v in server_state_dict.items() if 'bn' not in k}
            self.model.load_prompt(server_dict, strict=False)
        else:
            self.model.load_prompt(server_state_dict)



    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # _writer = glo.get_value("writer")
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            cnt = 0 
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                if self.args.debug and cnt>5:
                    break
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                if self.args.sam_mode == 'none':
                    self.optimizer.step()
                else:
                    self.optimizer.first_step(zero_grad=True)
                    log_probs = self.model(images)
                    self.criterion(log_probs, labels).backward()  # make sure to do a full forward pass
                    self.optimizer.second_step(zero_grad=True)
                batch_loss.append(loss.item())
                cnt+=1
                # logging.info('(client {} cnt {}'.format(self.client_index,cnt))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # self.writer.add_scalar('Loss/client_{}/train'.format(self.client_index), sum(batch_loss) / len(batch_loss), epoch)
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().obtain_prompt()
        return weights

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        img_size = 224 if '224' in args.vit_type else 384
        patch_size = 16 if '16' in args.vit_type else 32
        self.model = self.model_type(basic_model=server_dict['basic_model'], num_classes=self.num_classes,VPT_type=args.vpt_type,Prompt_Token_num=args.prompt_num,edge_size=img_size,patch_size=patch_size).to(self.device)
        wandb.watch(self.model)

    def start(self):
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write('{}\n'.format(self.args))
        return [self.model.cpu().obtain_prompt() for x in range(self.args.thread_number)]

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.obtain_prompt()
        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_prompt(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().obtain_prompt() for x in range(self.args.thread_number)]


if __name__ == '__main__':
    from main import add_args, allocate_clients_to_threads
    from methods import prompt
    from models.vpt import build_promptmodel
    import timm
    import argparse
    import data_preprocessing.data_loader as dl
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    img_size = 224 if '224' in args.vit_type else 384
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
    class_num = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size,img_size)

    mapping_dict = allocate_clients_to_threads(args)

    Server = prompt.Server
    Client = prompt.Client
    basic_model = timm.create_model(args.vit_type, num_classes= class_num, pretrained= True)
    Model = build_promptmodel

    server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'basic_model':basic_model}
    client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'basic_model':basic_model, 'num_classes': class_num} for i in range(args.thread_number)]
   
    client = Client(client_dict[0], args)