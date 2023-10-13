import os, re
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset.data_utils import get_file_name, get_csv_file, get_frame
import dataset.transform as transform

type = 'uint8'

class load_dataset(torch.utils.data.Dataset):
    def __init__(self, args : dict, split : str ='train', debug : bool =False):
        self.video_dirs = get_file_name(args['video_dirs'])
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])

        assert split in ['train', 'validation', 'test'], "The mode can only be 'train', 'validation' and 'test'"
        if split == 'train':
            self.transform = transforms.Compose([
                             transform.ToTensorVideo(),
                             transform.NormalizeVideo(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
                             transform.RandomResizedCropVideo(args['crop_size']),])
        if split == 'validation' or 'test':
            self.transform = transforms.Compose([
                             transform.ToTensorVideo(),
                             transform.NormalizeVideo(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),])

        self.label, self.key_index = get_csv_file(args['label_dir']+'/jester-v1-{}.csv'.format(split))

    def __len__(self):
        return len(self.key_index)

    def __getitem__(self, index):
        key_id = int(self.key_index[index])
        """ the dict of the label start from 1 while the video dirs start from 0
        That's why we need to use key_id-1 in here """
        item = self.process_frame(key_id-1)
        return item, self.label[str(key_id)]

    def process_frame(self, index):
        frames = np.zeros((self.sample_length, self.w, self.h, 3), np.dtype(type))
        imagelist = get_frame(self.video_dirs[index])
        for i, imagepath in enumerate(imagelist):
            if i == self.sample_length:
                break
            else:
                img = Image.open(imagepath).convert('RGB').resize(self.size)
                frames[i] = img
        release = -1

        if i < self.sample_length-1:
            frames[i:] = img
            release = i-1
        return self.transform(frames)
