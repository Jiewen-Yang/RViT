import os
import sys
import argparse
import time
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
from visualizer import get_local

from dataset.data_loader import load_dataset
from dataset.data_visual import visual_data

from tools.utils import Multi_Accuracy
from tools.init import Initialization
from tools.visualizer import *


os.environ['CUDA_ENABLE_DEVICES'] = '0'
frozen_list =['embedding.weight', 'embedding.bias']

parser = argparse.ArgumentParser()

# Dataset Config
parser.add_argument('--data_dir', type=str, default='/home/Dataset/Jester-V1/video',
                    help='the path of data')
parser.add_argument('--label_dir', type=str, default='/home/Dataset/Jester-V1/label',
                    help='the path of label')
parser.add_argument('--model_path', type=str, default=None,
                    help='the path of pretrained model')
parser.add_argument('--model_save_dir', type=str, default='./result/writer/',
                    help='the path of the trained model')
parser.add_argument('--writer_save_dir', type=str, default='./result/writer/',
                    help='the path of the writer')

# Model Config
parser.add_argument('--layer_num', type=int, default=2,
                    help='RViT layer numbers')
parser.add_argument('--class_num', type=int, default=27,
                    help='the number of action categories')
parser.add_argument('--sample_length', type=int, default=36,
                    help='the length of singal clip')
parser.add_argument('--image_size', type=tuple, default=(112, 112),
                    help="The size of image size, format is (int, int)")
parser.add_argument('--crop_size', type=int, default=112,
                    help="The size of image cropping size, format is int")
parser.add_argument('--patch_size', type=tuple, default=(8, 8),
                    help="The size of each patch")
parser.add_argument('--num_heads', type=int, default=8,
                    help="The number of heads in Multi-Head attention")
parser.add_argument('--embed_type', type=str, default='conv',
                    help="The embedding type we use, select 'conv' or 'net'")
parser.add_argument('--attn_type', type=str, default='linear',
                    help="The attention type we use, select 'linear', 'softmax' or 'reattn'")
parser.add_argument('--dropout', type=float, default=0.0,
                    help="The dropout rate RViT")

# Training Parameters
parser.add_argument('--train', action='store_true', default=False,
                    help='Use the training mode or validation mode')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='Do the attention map visualize')
parser.add_argument('--distributed', type=bool, default=False,
                    help='Using distributed training')
parser.add_argument('--enable_GPUs_id', type=list, default=[0], 
                    help = 'gpu devices ids')
parser.add_argument('--local_rank', type=int,
                    help='using which GPU(s)')
parser.add_argument('--layer_frozen', action='store_true', default=False, 
                    help = 'setting the type of optimizer',)
parser.add_argument('--batch_size', type=int, default=32,
                    help='size for each minibatch')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='maximum number of epochs')
parser.add_argument('--optimizer', type=str, help = 'setting the type of optimizer',
                    default='AdamW')
parser.add_argument('--loss_type', type=list, default=['LSCE'],
                    help='the type of criterion')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--warm_up', type=int, default=1,
                    help='warm up epoch')
parser.add_argument('--step_size', type=int, default=10,
                    help='initial step_size')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='initial betas param')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight_decay rate')
parser.add_argument('--clip_grad', action='store_true', default=False,
                    help='Using the clip gradient normlization')
parser.add_argument('--seed', type=int, default=8415,
                    help='seed for random initialisation')
parser.add_argument('--num_workers', type=int, help = 'setting the workers number',	
                    default=16)

args = parser.parse_args()


class Demo():
    def __init__(self, debug=False):
        torch.backends.cudnn.benchmark = True
        if 'WORLD_SIZE' in os.environ:
            args.distributed = int(os.environ['WORLD_SIZE']) > 1
        self.init = Initialization()
        self.device, self.local_rank = self.init.init_params(args.enable_GPUs_id, args.distributed)
        '''
        self.model = RViT(image_size = args.image_size[0],
                          patch_size = args.patch_size[0],
                          num_classes = args.class_num,
                          depth = args.layer_num,
                          length = args.sample_length,
                          heads = args.num_heads,
                          mlp_dim = args.patch_size[0] ** 2 * 3 * 4,
                          dropout = args.dropout,
                          embed_type = args.embed_type,
                          attn_type = args.attn_type)
        '''
        self.model = torch.jit.load(args.model_path)
        
        self.model = self.init.to_GPU(self.model, self.device, self.local_rank)
        print(self.model)
        
        if args.layer_frozen:
            self.model = self.init.frozen_layer(self.model, frozen_list)
        
        self.optimizer, self.scheduler = self.init.optimizer_init(self.model,
                                                                  args.optimizer,
                                                                  args.lr,
                                                                  args.step_size,
                                                                  args.betas,
                                                                  args.weight_decay)
        
        self.criterion = self.init.init_criterion(self.device, args.loss_type)
        self.model = self.init.use_multi_GPUs(self.model, self.local_rank, args.enable_GPUs_id, args.distributed)
        
        param = {'w':args.image_size[0], 
                 'h':args.image_size[1], 
                 'crop_size':args.crop_size, 
                 'sample_length':args.sample_length, 
                 'video_dirs':args.data_dir, 
                 'label_dir':args.label_dir}

        if args.train:
            self.train_dataset = load_dataset(args=param, split='train')

            if args.distributed:
                self.sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, 
                                                                               rank=self.local_rank,
                                                                               shuffle=True,)
            else:
                self.sampler = None

            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           shuffle=(self.sampler is None),
                                           drop_last=True,
                                           sampler=self.sampler)

        if args.visualize:
            self.val_dataset = visual_data(args=param, split='validation')
        else:
            self.val_dataset = load_dataset(args=param, split='validation')

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=1 if args.visualize else args.batch_size,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     shuffle=True,
                                     drop_last=True) if self.local_rank == args.enable_GPUs_id[0] else None

        self.summary, self.print_info, self.count, self.val_count = {}, {}, 0, 0
        self.writer = SummaryWriter(os.path.join(args.writer_save_dir, 'loss'))

        self.h_0 = torch.autograd.Variable(torch.zeros(args.layer_num, 
                                                       1 if args.visualize else args.batch_size, 
                                                       (args.image_size[0]//args.patch_size[0])**2, 
                                                       args.patch_size[0] ** 2 * 3, 
                                                       requires_grad=True)).to(self.device)

    def train(self):
        for self.epoch in range(args.num_epochs):
            
            if self.epoch < args.warm_up:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = args.lr * (10 ** (self.epoch-args.warm_up+1))
            
            print("current learning rate : ", param_group['lr'])
            print('Start Epoch: {}'.format(self.epoch))
            
            self.model.train()
            for self.step, (inputs, labels) in enumerate(tqdm(self.train_loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                result, _ = self.model(inputs.view(-1, 3, args.image_size[0], args.image_size[1]), self.h_0)
                loss = self.criterion[0](result, labels)

                top1, top5 = Multi_Accuracy(result.data, labels, topk=(1, 5))

                if self.local_rank == args.enable_GPUs_id[0]:
                    self.count += 1
                    self.add_summary(self.writer, 'train/loss', loss.item())
                    self.add_summary(self.writer, 'train/top1', top1.item())
                    self.add_summary(self.writer, 'train/top5', top5.item())

                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        tag = tag.replace('module', '')
                        self.add_summary(self.writer, tag, value.data.cpu().numpy(), sum_type='histogram')

                self.optimizer.zero_grad()
                
                loss.backward(retain_graph=True)
                if args.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                
                self.optimizer.step()

            self.scheduler.step()

            if self.local_rank == args.enable_GPUs_id[0]:
                print('------Training Result------\n \
                       Top-1 accuracy: {top1_acc:.2f}%, \
                       Top-5 accuracy: {top5_acc:.2f}% \
                       Loss value: {loss:.2f}'.\
                       format(top1_acc=self.print_info['train/top1'], 
                              top5_acc=self.print_info['train/top5'], 
                              loss=self.print_info['train/loss'],)
                              )
                print('End Training Epoch: {}'.format(self.epoch))

            self.model.eval()
            self.validation()
            self.save()

    def validation(self):
        total_top1, total_top5 = 0, 0
        if self.local_rank == args.enable_GPUs_id[0]:
            with torch.no_grad():
                for self.step, (inputs, labels) in enumerate(tqdm(self.val_loader)):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    result, _ = self.model(inputs.view(-1, 3, args.image_size[0], args.image_size[1]), self.h_0)
                    loss = self.criterion[0](result, labels)

                    top1, top5 = Multi_Accuracy(result.data, labels, topk=(1, 5))

                    self.val_count += 1
                    self.add_summary(self.writer, 'val/loss', loss.item(), record_type = 'validation')
                    self.add_summary(self.writer, 'val/top1', top1.item(), record_type = 'validation')
                    self.add_summary(self.writer, 'val/top5', top5.item(), record_type = 'validation')

                    total_top1 += top1.item()
                    total_top5 += top5.item()

            print('------Validation Result------\n \
                   Top-1 accuracy: {top1_acc:.2f}%, \
                   Top-5 accuracy: {top5_acc:.2f}% \
                   Loss value: {loss:.2f}'.\
                    format(top1_acc=total_top1/self.step,
                           top5_acc=total_top5/self.step,
                           loss=self.print_info['val/loss'],))
        else:
            pass

    def visualize(self):
        with torch.no_grad():
            for self.step, (inputs, labels, frame_list) in enumerate(tqdm(self.val_loader)):
                #inputs, labels, redundance_label = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                result, _ = self.model(inputs.view(-1, 3, args.image_size[0], args.image_size[1]), self.h_0)
                cache = get_local.cache
                if self.step == 0:
                    print(cache.keys())
                    attention_maps = cache['MultiHeadDotProductAttention.linear_attn']
                    vid_len = len(frame_list)
                    for i in range(1,vid_len*2,2):
                        #visualize_grid_to_grid_with_cls(attention_maps[i][0,h,:,:], 105, np.array(frame_list[i//2][0]), num=i, head=h)
                        #visualize_grid_to_grid(attention_maps[i][0,h,1:,1:], 100, np.array(frame_list[i//2][0]), num=i, head=h)
                        for j in range(args.num_heads):
                            attention_map = np.mean(attention_maps[i][0,:,:,:],axis=0,keepdims=False)
                            visualize_grid_to_grid_with_cls(attention_map, 98, np.array(frame_list[i//2][0]),alpha=0.55, num=i, head=j)

                        visualize_heads(attention_maps[i],cols=4,num=i//2)


    # add summary
    def add_summary(self, writer, name, val, sum_type = 'scalar', record_type = 'train'):
        def writer_in(writer, name, val, sum_type, count):
            if sum_type == 'scalar':
                writer.add_scalar(name, self.summary[name]/100, count)
                self.print_info[name] = self.summary[name]/100
                self.summary[name] = 0
            elif sum_type == 'image':
                writer.add_image(name, val, count)
            elif sum_type == 'histogram':
                writer.add_histogram(name, val, count)

        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.count % 100 == 0 and record_type == 'train':
            writer_in(writer, name, val, sum_type, self.count)
        elif writer is not None and self.val_count % 100 == 0 and record_type == 'validation':
            writer_in(writer, name, val, sum_type, self.val_count)

    def save(self):
        torch.save({
            'net': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(args.model_save_dir, 'model_RTrans_{}.pth'.format(self.epoch)))


def main():
    process = Demo()
    if args.train:
        process.train()
    elif args.visualize:
        get_local.activate()
        process.visualize()
    else:
        process.validation()

if __name__ == "__main__":
    main()