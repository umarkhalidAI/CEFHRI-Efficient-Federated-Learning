import torch
import wandb
import logging
from methods.base import Base_Client, Base_Server
# from data_preprocessing.dpsgd import DPSGD

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)

        self.model = self.model_type(client_dict['type'], num_classes=self.num_classes, pretrained=False).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        params = filter(lambda p: p.requires_grad,self.model.parameters())


        if args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        elif args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(server_dict['type'], num_classes=self.num_classes, pretrained=False).to(self.device)