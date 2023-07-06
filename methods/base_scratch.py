import torch
import wandb
import logging

logging.basicConfig(level=logging.INFO)
from torch.multiprocessing import current_process
from torch.nn.utils import clip_grad_norm_
# from torch.utils.tensorboard import SummaryWriter
# import writer
import numpy as np
import util.lr_sched as lr_sched
from util import misc
from timm.utils import accuracy


def get_noise_multiplier(epsilon, delta, max_grad_norm):
    """
        grads: [N, d]
    """

    # # sensitivity
    s = 2 * max_grad_norm

    sigma = s * np.sqrt(2 * np.log(1.25 / delta) / epsilon)

    return sigma


class Base_Client():
    def __init__(self, client_dict, args):
        self.train_data = client_dict['train_data']
        self.test_data = client_dict['test_data']
        self.device = 'cuda:{}'.format(client_dict['device'])
        if 'model_type' in client_dict:
            self.model_type = client_dict['model_type']
        elif 'model' in client_dict:
            self.model = client_dict['model']
        # self.writer = SummaryWriter(args.save_path)
        self.tuning_config = client_dict['tuning_config']
        self.num_classes = client_dict['num_classes']
        self.args = args
        self.round = 0
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None
        # self.epochs=0

    def set_server(self, server):
        self.server = server

    def load_client_state_dict(self, server_state_dict):
        if self.args.localbn:
            server_dict = {k: v for k, v in server_state_dict.items() if 'bn' not in k}
            self.model.load_state_dict_adapt(server_dict, strict=False)
        else:
            # for k, v, in server_state_dict:
            #     print (k)

            self.model.load_state_dict_adapt(server_state_dict)

    def run(self, received_info):
        client_results = []
        # try:
        # print(self.client_map)
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            # print(num_samples, "Samples")

            # if self.args.dp:
            self.noise_multiplier = get_noise_multiplier(self.args.epsilon, self.args.delta, self.args.max_grad_norm)
            # logging.info('noise_multiplier:{}'.format(self.noise_multiplier))

            weights, train_acc = self.train(self.round)  # if not self.args.dp else self.dp_train()
            acc = self.test()
            # self.model.to('cpu')
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            client_results.append({'weights': weights, 'num_samples': num_samples, 'acc': acc, 'train_acc': train_acc,
                                   'client_index': self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()
        # except:
        #     print(self.client_index, self.round)

        self.round += 1
        return client_results

    def train(self, round):
        eff_batch_size = self.args.batch_size * self.args.accum_iter * misc.get_world_size()
        starting_epoch = round * self.args.epochs
        end_epoch = starting_epoch + self.args.epochs

        if self.args.lr is None:  # only base_lr is specified
            self.args.lr = self.args.blr * eff_batch_size / 256
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # _writer = glo.get_value("writer")
        epoch_loss = []
        max_accuracy = 0.0
        # print("Starting Training")
        train_sample_number = train_correct = 0
        # max_epoch=self.epochs+self.args.local_epochs
        for epoch in range(starting_epoch, end_epoch):
            accum_iter = self.args.accum_iter  # Need
            batch_loss = []
            cnt = 0
            self.optimizer.zero_grad()

            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # if batch_idx % accum_iter == 0:
                #     lr_sched.adjust_learning_rate(self.optimizer, batch_idx / len(self.train_dataloader) + epoch, self.args)
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                # if batch_idx % self.args.accum_iter == 0:
                #     lr_sched.adjust_learning_rate(self.optimizer, batch_idx / len(self.train_dataloader) + epoch,
                #                                   self.args)
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss_value = loss.item()
                    loss /= accum_iter
                    _, predicted = torch.max(outputs, 1)
                    correct = predicted.eq(labels).sum()

                    train_correct += correct.item()
                    # test_loss += loss.item() * target.size(0)
                    train_sample_number += labels.size(0)
                    cnt += 1
                    self.loss_scaler(loss, self.optimizer, clip_grad=None,
                                     parameters=self.model.parameters(), create_graph=False,
                                     update_grad=(batch_idx + 1) % accum_iter == 0, args=self.args,
                                     last_epoch=end_epoch - epoch, model=self.model,
                                     noise_multiplier=self.noise_multiplier, device=self.device)
                    # self.optimizer.step()
                    if (batch_idx + 1) % accum_iter == 0:
                        self.optimizer.zero_grad()
                    torch.cuda.synchronize()
            train_acc = (train_correct / train_sample_number) * 100
            # print("client: ", self.client_index, "epoch: ", epoch, "train_Acc: ", train_acc)
            batch_loss.append(loss_value)
            cnt += 1
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # self.writer.add_scalar('Loss/client_{}/train'.format(self.client_index), sum(batch_loss) / len(batch_loss), epoch)
                # logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                #                                                             epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
                print(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch,
                                                                                                    sum(epoch_loss) / len(
                                                                                                        epoch_loss),
                                                                                                    current_process()._identity[
                                                                                                        0],
                                                                                                    self.client_map[
                                                                                                        self.round]))

        weights = self.model.cpu().state_dict_adapt()
        # images, labels = images.to('cpu'), labels.to('cpu')

        return weights, train_acc

    def dp_train(self):
        misc.load_model(args=self.args, model_without_ddp=self.model, optimizer=self.optimizer,
                        loss_scaler=self.loss_scaler)
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # _writer = glo.get_value("writer")
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            cnt = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                if self.args.debug and cnt > 5:
                    break
                for param in self.model.parameters():
                    param.accumulated_grads = []

                images, labels = batch

                for image, label in zip(images, labels):
                    image, label = image.to(self.device).unsqueeze(0), label.to(self.device).unsqueeze(0)

                    self.optimizer.zero_grad()
                    log_probs = self.model(image)
                    loss = self.criterion(log_probs, label)
                    batch_loss.append(loss.item())
                    loss.backward()

                    for param in self.model.parameters():
                        per_sample_grad = param.grad.detach().clone()
                        clip_grad_norm_(per_sample_grad, max_norm=self.args.max_grad_norm)  # in-place
                        param.accumulated_grads.append(per_sample_grad)
                for param in self.model.parameters():
                    param.grad = torch.stack(param.accumulated_grads, dim=0)

                for param in self.model.parameters():
                    param = param - self.args.lr * param.grad
                    param += torch.normal(mean=0, std=self.noise_multiplier * self.args.max_grad_norm)

                    param.grad = 0  # Reset for next iteration

                cnt += 1
                # logging.info('(client {} cnt {}'.format(self.client_index,cnt))
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    # self.writer.add_scalar('Loss/client_{}/train'.format(self.client_index), sum(batch_loss) / len(batch_loss), epoch)
                    logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(
                        self.client_index,
                        epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0],
                        self.client_map[self.round]))
                    # print('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(
                    #     self.client_index,
                    #     epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0],
                    #     self.client_map[self.round]))
        weights = self.model.cpu().state_dict_adapt()
        return weights

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            cnt = 0
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                # if self.args.debug and cnt>1:
                #     break
                x = x.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                # with torch.cuda.amp.autocast():
                pred = self.model(x)

                # acc1, acc5 = accuracy(pred, target, topk=(1, 5))
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                cnt += 1
            acc = (test_correct / test_sample_number) * 100
            # logging.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
            # print("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))

        return acc


class Base_Server():
    def __init__(self, server_dict, args):
        self.train_data = server_dict['train_data']
        self.test_data = server_dict['test_data']
        self.device = 'cuda:{}'.format(torch.cuda.device_count() - 1)
        # self.device = 'cpu'
        if 'model_type' in server_dict:
            self.model_type = server_dict['model_type']
        elif 'model' in server_dict:
            self.model = server_dict['model']
        self.num_classes = server_dict['num_classes']
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.save_path = server_dict['save_path']
        self.tuning_config = server_dict['tuning_config']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        try:
            # self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
            # self.device = 'cuda:0'
            acc, loss = self.test()
            # self.model.to('cpu')
            # self.device = 'cpu'
            with torch.cuda.device('cuda:1'):
                torch.cuda.empty_cache()
        except:
            logging.info("Now using cpu for Server.")
            self.device = 'cpu'
            acc = self.test()

        self.log_info(received_info, acc, loss=loss)
        self.round += 1
        if acc > self.acc:
            if self.args.save_model:
                torch.save(self.model.state_dict_adapt(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        return server_outputs

    def start(self):
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write('{}\n'.format(self.args))
        return [self.model.cpu().state_dict_adapt() for x in range(self.args.thread_number)]

    def log_info(self, client_info, acc, loss):
        client_acc = sum([c['acc'] for c in client_info]) / len(client_info)
        client_tacc = sum([c['train_acc'] for c in client_info]) / len(client_info)
        if not self.args.debug:
            wandb.log({"Test/AccTop1": acc, "Test_Loss": loss, "Client_Train/AccTop1": client_tacc,
                       "Client_Test/AccTop1": client_acc, "round": self.round})
        out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(acc, client_acc, self.round)
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write(out_str)

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        #print(len(client_info))
        #exit()
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]
        ssd = self.model.state_dict_adapt()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict_adapt(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict_adapt() for x in range(self.args.thread_number)]

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            cnt = 0
            for batch_idx, (x, target) in enumerate(self.test_data):
                if self.args.debug and cnt > 1:
                    break
                x = x.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                # with torch.cuda.amp.autocast():
                # output = model(images)
                pred = self.model(x)
                # acc1, acc5 = accuracy(pred, target, topk=(1, 5))
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                predicted = predicted.to(target.device)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                cnt += 1
            acc = (test_correct / test_sample_number) * 100
            logging.info("************* Server Acc = {:.2f} **************".format(acc))
            # print("************* Server Acc = {:.2f} **************".format(acc))

        # self.device = 'cpu'
        # x = x.to('cpu')
        # target = target.to('cpu')

        return acc, test_loss
