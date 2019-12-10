import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import sampler
import six
import sys
from PIL import Image
import numpy as np


class lmdbDataset(Dataset):
    """
        New Dataloader Test Class
    """ 
    def __init__(self, rootfile, transform=None, reverse=False, mode=True, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
        self.rootfile = rootfile
        self.transform = transform
        self.alphabet = alphabet
        self.reverse = reverse
        self.listIdx = []
        self.img_loc = []
        self.text = []
        split = 0.75

        with open(rootfile, 'r') as f:
            self.listIdx = f.readlines()

        if mode:
            self.listIdx = self.listIdx[:int(split * len(self.listIdx))]
        else:
            self.listIdx = self.listIdx = self.listIdx[int(split * len(self.listIdx)):]

        for i in range(len(self.listIdx)):
            loc, txt = self.listIdx[i].split('\n')[0].split(':')
            self.img_loc.append(loc)
            self.text.append(txt)

    def __len__(self):
        return len(self.listIdx)

    def __getitem__(self, idx):
        img = Image.open(self.img_loc[idx]).convert('L')
        label = self.text[idx]
        label = ''.join(label[i] if label[i].lower() in self.alphabet else ''
                for i in range(len(label)))
        if len(label) <= 0:
            return self[idx + 1]
        if self.reverse:
            label_rev = label[-1::-1]
            label_rev += '$'
        label += '$'

        if self.transform is not None:
            img = self.transform(img)

        if self.reverse:
            return (img, label, label_rev)
        else:
            return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)