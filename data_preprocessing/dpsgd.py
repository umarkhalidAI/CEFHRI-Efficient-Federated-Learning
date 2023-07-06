#  Copy from https://github.com/srxzr/DPSGD/blob/master/dpsgd.py
import math
import torch

import uuid
import torch
from torch.optim.optimizer import Optimizer, required

import torch.nn.functional as F

class DPSGD(Optimizer):


    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,C=1,noise_multiplier= 1.0 , batch_size=256):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DPSGD, self).__init__(params, defaults)

        
        self.batch_size = batch_size
        
        self.C = C
        self.bigger_batch = {}


        self.bigger_batch_count = {}
        self.noise_multiplier = noise_multiplier
        
    def __setstate__(self, state):
        super(DPSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        noise_std =self.noise_multiplier *self.C
        
        loss = None
        if closure is not None:
            loss = closure()

        
        norm  = 0
        
            
        for group in self.param_groups:
        
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                
                if not hasattr(p,'myid'):
                    p.myid = uuid.uuid4()
                    self.bigger_batch[p.myid] = torch.zeros_like(grad)
                    
                    self.bigger_batch_count[p.myid] =  torch.cuda.LongTensor(size=[1]).zero_() if self.bigger_batch[p.myid].is_cuda else  torch.LongTensor(size=[1]).zero_()

                norm+=grad.norm()**2.0
        norm=norm**(0.5)
                
        
        for group in self.param_groups:
        
            for p in group['params']:
                grad = p.grad.data
                
                
                cliped = (grad*self.C) /( torch.max(norm,torch.ones_like(norm)*self.C))
                self.bigger_batch [p.myid].add_(cliped)
                self.bigger_batch_count[p.myid]+=1
                

        if   self.bigger_batch_count[p.myid] == self.batch_size:


            for group in self.param_groups:
            
                for p in group['params']:
                    
                    

                    
                    base =  self.bigger_batch[p.myid]
                    my_rand = torch.zeros_like(base) 
                    my_rand.normal_(mean=0, std =noise_std  )
                    
                    base.add_(my_rand)
                
                    base = base/float(self.batch_size) 

                    self.bigger_batch[p.myid] =base
                    

                    
                   
                                    
            
            


            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = self.bigger_batch[p.myid]
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)
                    self.bigger_batch[p.myid].zero_()
                    self.bigger_batch_count[p.myid].zero_()

        return loss