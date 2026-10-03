
class LearningRateScheduler(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def step(self, *args, **kwargs):
        raise NotImplementedError
        
    @staticmethod
    def set_lr(optimizer, lr):
        if lr > 0:
            optimizer.param_groups[0]["lr"] = lr
        
    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state):
        self.__dict__.update(state)

class WarmupLRScheduler(LearningRateScheduler):
    def __init__(self, optimizer, min_lr: float, max_lr :float, warmup_steps: int):
        super(WarmupLRScheduler, self).__init__(optimizer)
        self.min_lr = min_lr # 1e-8
        self.max_lr = max_lr # 1e-3
        self.warmup_steps = warmup_steps

        if warmup_steps != 0:
            warmup_rate = self.max_lr - self.min_lr
            self.warmup_rate = warmup_rate / warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 0

    def step(self):
        self.lr = self.get_lr()
        if self.lr < self.max_lr:
            if self.update_steps < self.warmup_steps-1:
                self.lr = self.min_lr + self.warmup_rate * self.update_steps
                self.set_lr(self.optimizer, self.lr)
                self.update_steps += 1
            elif self.update_steps == self.warmup_steps - 1:
                self.set_lr(self.optimizer, self.max_lr)
        return self.lr

class Poly(LearningRateScheduler):
    def __init__(self,   optimizer, 
                         start_epoch : int, 
                         max_epochs : int, 
                         exponent : float = 0.9, 
                         min_lr: float =1e-4, 
                         initial_lr : float=1e-3):
        super(Poly, self).__init__(optimizer)
        self.initial_lr = initial_lr
        self.waiting_epoch = start_epoch
        self.max_epochs = max_epochs - self.waiting_epoch
        self.exponent = exponent
        self.min_lr = min_lr

    def step(self, epoch = None):
        lr = self.get_lr()
        if epoch is not None and (epoch+1 >= self.waiting_epoch) and lr > self.min_lr:
            lr = self.initial_lr * (1 - (epoch+1 - self.waiting_epoch) / self.max_epochs)**self.exponent
            if isinstance(lr, float) and lr > self.min_lr:
                self.set_lr(self.optimizer, lr)
            else:
                self.set_lr(self.optimizer, self.min_lr)
        return lr

class WarmupScheduler(LearningRateScheduler):
    def __init__(self, optimizer, 
                 min_lr: float, 
                 max_lr: float, 
                 warmup_steps: int, 
                 after_warmup_scheduler = None):
        super(WarmupScheduler, self).__init__(optimizer)
        assert after_warmup_scheduler is None or isinstance(after_warmup_scheduler, ReduceLROnPlateau) or \
                                                                isinstance(after_warmup_scheduler, Poly), \
                                            f"{after_warmup_scheduler} is not supported after Warmup steps"
        self.warmup_steps = warmup_steps
        self.update_steps = 0
        Wscheduler = WarmupLRScheduler(optimizer = optimizer,
                                       min_lr = min_lr,
                                       max_lr = max_lr,
                                       warmup_steps = warmup_steps)
        self.schedulers = [Wscheduler]
        if after_warmup_scheduler is not None:
            self.schedulers.append(after_warmup_scheduler)

    def state_dict(self):
        # plain values only: the scheduler objects hold the optimizer, and checkpoints must load with weights_only
        return {**super().state_dict(), 'schedulers': [scheduler.state_dict() for scheduler in self.schedulers]}

    def load_state_dict(self, state):
        for scheduler, scheduler_state in zip(self.schedulers, state['schedulers']):
            scheduler.load_state_dict(scheduler_state)
        super().load_state_dict({key: value for key, value in state.items() if key != 'schedulers'})

class ReduceLROnPlateau(LearningRateScheduler):
    def __init__(self, optimizer, patience = 5, factor = 0.5, wloss=False, start_epoch=0):
        super(ReduceLROnPlateau, self).__init__(optimizer)
        self.start_epoch = start_epoch
        self.patience = patience
        self.factor = factor
        self.wloss = wloss
        
        self.loss = float("inf")
        self.count = 0

    def step(self, loss = None, epoch=None):
        self.lr = self.get_lr()
        if epoch is not None and epoch >= self.start_epoch and self.wloss:
            self.count += 1
            if self.patience == self.count:
                self.count = 0
                self.patience = self.patience // 1.2
                self.lr *= self.factor
                self.set_lr(self.optimizer, self.lr)
        return self.get_lr()