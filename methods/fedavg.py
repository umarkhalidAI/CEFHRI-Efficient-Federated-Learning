import torch
import wandb
import logging
from methods.base import Base_Client, Base_Server

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        if 'model_type' in client_dict:
            self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        if 'model_type' in server_dict:
            self.model = self.model_type(self.num_classes).to(self.device)

