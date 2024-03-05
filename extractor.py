import os
import torch
from torch.autograd import Variable as V


class Extractor:
    def __init__(self, net, loss=None, eval_mode=False, lr=0.001, step_size=10):
        self.eval_mode = eval_mode
        self.net = torch.nn.DataParallel(net(eval_mode=self.eval_mode).cuda(), device_ids=range(torch.cuda.device_count()))

        if eval_mode:
            self.net.eval()
        else:
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
            self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.5)
            self.loss = loss()

        self.sat = None
        self.pre = None
        self.gt = None

    def set_input(self, data_batch):
        self.sat = V(data_batch['sat'].cuda())
        self.gt = {'seg': V(data_batch['seg'].cuda()), 'vex': V(data_batch['vex'].cuda()), 'ori': V(data_batch['ori'].cuda())}

    def optimize(self):
        self.optimizer.zero_grad()
        self.pre = self.predict(self.sat)
        loss = self.loss(self.pre, self.gt)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, sat):
        return self.net.forward(sat)

    def save(self, path, epoch_id):
        torch.save({
            'epoch_id': epoch_id,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exp_lr_scheduler_state_dict': self.exp_lr_scheduler.state_dict()
        }, os.path.join(path))

    def load(self, path):
        if os.path.exists(os.path.join(path)):
            checkpoint = torch.load(os.path.join(path))
            model_dict = self.net.state_dict()
            state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.net.load_state_dict(model_dict)
            if not self.eval_mode:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.exp_lr_scheduler.load_state_dict(checkpoint['exp_lr_scheduler_state_dict'])
            return checkpoint['epoch_id'] + 1
        else:
            return 1

    def update_learning_rate(self):
        self.exp_lr_scheduler.step()

    def lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def visual(self):
        return self.sat, self.pre
