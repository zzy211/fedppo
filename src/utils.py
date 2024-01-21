#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, iid_sampling, non_iid_dirichlet_sampling
import numpy as np
from PIL import Image
import os
from keras.datasets import imdb 
from keras.preprocessing.sequence import pad_sequences 
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    
    if args.dataset == 'cifar10':
        data_dir = '../data/cifar10' #在父附录中(..)找到data/cifar
        args.num_classes = 10
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_val)
        n_train = len(train_dataset)
        y_train = np.array(train_dataset.targets)
        # sample training data amongst users
        if args.iid:
            #sample IID user data from cifar
            user_groups = iid_sampling(n_train, args.num_users, args.seed)
        else:
            user_groups = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
        
    elif args.dataset == 'cifar100':
        data_dir = '../data/cifar100'
        # print("data_dir:",data_dir)
        args.num_classes = 100
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_val)
        n_train = len(train_dataset)
        y_train = np.array(train_dataset.targets)
        # sample training data amongst users
        if args.iid:
            #sample IID user data from cifar
            user_groups = iid_sampling(n_train, args.num_users, args.seed)
        else:
            user_groups = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)


    elif args.dataset == 'clothing1m':
        # data_dir = '../data/clothing1m/'
        data_dir = '../data/clothing1m/'
        args.num_classes = 14
        args.model = 'resnet50'
        #对图像进行标准化处理
        trans_train = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trans_val = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 9.406], std=[0.229, 0.224, 0.225])
        ])
        #这里下午是否可以做修改，直接噪声率直接混合数据集
        train_dataset = Clothing(data_dir, trans_train, "train", args.level_n_system)
        test_dataset = Clothing(data_dir, trans_val, "test", args.level_n_system)
        n_train = len(train_dataset)
        y_train = np.array(train_dataset.targets)
        if args.iid:
            #sample IID user data from cifar
            user_groups = iid_sampling(n_train, args.num_users, args.seed)
        else:
            user_groups = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
    
    #尝试导入emnist数据集
    elif args.dataset == 'emnist':
        data_dir = '../data/emnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.EMNIST(data_dir, split="byclass", download=True, 
                                        train=True, transform=apply_transform)
        test_dataset = datasets.EMNIST(data_dir, split="byclass", download=True,
                                        train=False, transform=apply_transform)
        n_train = len(train_dataset)
        y_train = np.array(train_dataset.targets)
     # sample training data amongst users
        if args.iid:
            #sample IID user data from cifar
            user_groups = iid_sampling(n_train, args.num_users, args.seed)
        else:
            user_groups = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)


    elif args.dataset == 'mnist' or args.dataset ==  'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    
    elif args.dataset == 'imdb':
        print("!!!!!!!!数据集是imdb")
        MAX_WORDS = 10000 
        MAX_LEN = 200 
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
        x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
        x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
        n_train = x_train.shape
        target_train = np.array(y_train)
        args.num_classes = 2
        # sample training data amongst users
        if args.iid:
            #sample IID user data from cifar
            user_groups = iid_sampling(n_train, args.num_users, args.seed)
        else:
            user_groups = non_iid_dirichlet_sampling(target_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
        return x_train, y_train, x_test, y_test, user_groups
    
    else:
        exit('Error:unrecognized dataset')

    return train_dataset, test_dataset, user_groups


def average_weights(w, dataset_list):   #dataset_list中一次存储了每个参与训练的client的数据集大小
    """
    Returns the average of the weights.
    """
    #计算每个client对应的权重 dk/D
    w_dataset = np.array(dataset_list)/np.sum(np.array(dataset_list))

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        weight_sum_key = torch.zeros_like(w_avg[key], dtype=torch.float)
        for i in range(0, len(w)):
            # weight_sum_key += w[i][key] * w_dataset[i]
            weight_sum_key += torch.mul(w[i][key], w_dataset[i])
        w_avg[key] = weight_sum_key
    return w_avg
        

    # w_avg = copy.deepcopy(w[0]) 
    # for key in w_avg.keys():    #循环遍历权重列表w中的所有元素
    #     for i in range(1, len(w)):  #循环遍历所有的client
    #         w_avg[key] += w[i][key] #将他们的key对应的权重值加起来
    #     w_avg[key] = torch.div(w_avg[key], len(w))
    # return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    dataset   : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode, noise_rate):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.noise_rate = noise_rate
        self.noisy_labels = {}  #将图像路径和噪声标签存储到根目录路径
        self.clean_labels = {}
        self.data = []  #用来存放图像数据
        self.targets = []


        with open(self.root + 'noisy_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.noisy_labels[img_path] = int(entry[1])
        
        with open(self.root + 'clean_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.clean_labels[img_path] = int(entry[1])
        
        if self.mode == 'train':    #此时加载的数据集本身就是混合了噪声的，所以理论上所有客户端都是噪声客户端
            with open(self.root + 'clean_train_key_list.txt', 'r') as f:
                lines_clean = f.read().splitlines()
            n_clean = len(lines_clean)
            for l in lines_clean:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.clean_labels[img_path]
                self.targets.append(target)
            n_noise_need = 100000 - n_clean  #认为总样本数是100000
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines_noise = f.read().splitlines()
            n_noise = len(lines_noise)
            np.random.seed(13)
            subset_idx = np.random.choice(n_noise, n_noise_need, replace=False)
            for i in subset_idx:
                l = lines_noise[i]
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)   

        elif self.mode == 'minitrain':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            n = len(lines)
            np.random.seed(13)
            subset_idx = np.random.choice(n, int(n/10), replace=False)
            for i in subset_idx:
                l = lines[i]
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'test':
            with open(self.root + 'clean_test_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.clean_labels[img_path]
                self.targets.append(target)
    
    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target
    
    def __len__(self):
        return len(self.data)
            

            

