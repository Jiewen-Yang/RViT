import os
#import apex

import torch
from torch import nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class Initialization:
    """docstring for initialization"""
    def __init__(self, local_rank = None):
        self.local_rank = None

    @staticmethod
    def init_params(enable_GPUs_id, distributed):
        if distributed:
            # FOR DISTRIBUTED:  Set the device according to local_rank.
            # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
            # environment variables, and requires that you use init_method=`env://`.
            torch.distributed.init_process_group(backend='nccl', 
                                                init_method='env://')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            torch.distributed.barrier()
            print("We would use distributed setting") if local_rank == enable_GPUs_id[0] else None

        else:
            print("We would use default setting")
            device = torch.device("cuda", enable_GPUs_id[0])
            local_rank = enable_GPUs_id[0]

        return device, local_rank

    
    @staticmethod
    def init_criterion(device, criterion_type):
        # you can defind your own criterion function here
        criterions = {"BCE":nn.BCEWithLogitsLoss(), 
                      "CE":nn.CrossEntropyLoss(),
                      "MSE":nn.MSELoss(reduction='none'),
                      "LSCE":LabelSmoothingCrossEntropy(),}

        return [criterions[cr].to(device) for cr in criterion_type]

    @staticmethod
    def optimizer_init(model, optimizer_name, lr, step_size, betas_param, weight_decay, scheduler_mode='step'):
        def optimizers(model, lr, betas_param, weight_decay):
            # you could define your own optimizers here

            SGD = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr, momentum=betas_param[0], weight_decay=weight_decay)
            Adam = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr, betas=betas_param)
            AdamW = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=lr, betas=betas_param)

            optimizer_dict = {"SGD": SGD, "Adam": Adam, "AdamW": AdamW}

            return optimizer_dict

        if optimizer_name in ["SGD", "Adam", "AdamW"]:
            optimizer = optimizers(model, lr, betas_param, weight_decay)[optimizer_name]
        else:
            raise Exception(
                "Only 'SGD', 'Adam' and 'AdamW' are available, if you want to use other optimizer, please add it in funtion optimizer_init() in init.py")

        if scheduler_mode=='cosineAnn':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
        elif scheduler_mode=='cosineAnnWarm':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)

        return optimizer, scheduler

    @staticmethod
    def load_pretrained_model(model, pretrained_path):
        if os.path.exists(pretrained_path):
            print("Loading Pretrained Model From : {}".format(pretrained_path))
            model_dict = model.state_dict()
            model_Checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
            optim_dict = model_Checkpoint['optimizer_state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in model_Checkpoint['net'].items() if k.replace('module.', '') in model_dict}
            print('Total : {}, update: {}'.format(len(model_Checkpoint['net']), len(pretrained_dict)))
        else:
            print("We could not find the model in path --> ", pretrained_path)

        return pretrained_dict, optim_dict

    @staticmethod
    def frozen_layer(model, frozen_list):             
        for name, value in model.named_parameters():
            if name in frozen_list:
                value.requires_grad = False 
        return model


    @staticmethod
    def to_GPU(model, device, local_rank ,mode="train"):
        if torch.cuda.is_available():
            print("Using GPU -> ", local_rank)
            return model.to(device)
        else:
            raise Exception("Cuda is Unavailable now, Please Check Your Device Setting")


    @staticmethod
    def use_multi_GPUs(model, local_rank, enable_GPUs_id, distributed=True): 
        def Distributed_GPUs(model, local_rank):
            torch.distributed.barrier()
            print("Using Multi-Gpu, devices count is:{}".format(torch.cuda.device_count())) if local_rank == enable_GPUs_id[0] else None
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)
            return model
        
        def Parallel_GPUs(model, local_rank):
            model = nn.DataParallel(model, device_ids=enable_GPUs_id)
            print("Using Multi-Gpu, device_ids is: {}".format(enable_GPUs_id))
            return model

        def Single_GPU():
            return print("Using Single-Gpu, device_ids is: {}".format(enable_GPUs_id[0]))

        if distributed:# multi-Gpu
            model = Distributed_GPUs(model, local_rank)
        elif distributed == False and len(enable_GPUs_id) > 1:
            model = Parallel_GPUs(model, local_rank)
        else:
            Single_GPU()

        return model
