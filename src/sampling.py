#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import copy
from numpy.testing import assert_array_almost_equal
from options import args_parser

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.train_labels)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)    #每次运行时生成相同的随机数序列
    num_items = int(n_train/num_users) #每个client分配num_items条数据
    dict_users, all_idxs = {}, [i for i in range(n_train)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs)-dict_users[i])
    return dict_users

def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100): #y_train是样本的标签，num_users是用户的数量，p是从伯努利分布中采样的概率
    np.random.seed(seed)
    #indicate the classes chosen by each client
    Phi = np.random.binomial(1, p , size=(num_users, num_classes))
    n_classes_per_client = np.sum(Phi, axis=1)
    while(np.min(n_classes_per_client)==0):
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    #indicate the clients that choose each class
    #Psi[j]表示选择了第j个类别的client列表
    Psi = [list(np.where(Phi[:,j]==1)[0]) for j in range(num_classes)]
    #num_clients_per_class[j]表示选择了第j个类别的client个数
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    dict_class = [[] for _ in range(num_users)] #dict_class[i]存储client i的所有样本的编号
    for class_i in range(num_classes):
        #找出标签为class_i的所有样本的id
        all_idxs = np.where(y_train==class_i)[0] #找到类别i的所有样本
        #选择了class_i样本的client可能得到该样本的概率
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        #为每个样本分配他们所选的client
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())
        for client_id in assignment:
            dict_class[client_id].append(class_i)
        for client_k in Psi[class_i]:
            if(client_k in dict_users):
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment==client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment==client_k)])
    
    #输出每个client拥有的sample的条数
    for a in range(0,len(dict_class)):
        print("client_{}_sample_num:{}".format(a, len(dict_class[a])))


    # print("以下是结果：")
    # result = [] #result[i]对应client i 每个类的个数
    # for client in range(num_users):
    #     client_result = []
    #     for class_id in range(0, num_classes):
    #         num = dict_class[client].count(class_id)
    #         client_result.append(num) #依次添加每一类的个数
    #     result.append(client_result)
    #     print(client_result)
    # print("result:",result)
    return dict_users

def multiclass_noisify(y, P, random_state=0):
    assert P.shape[0] == P.shape[1] #用来判断P是否是一个方阵
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1])) #计算P矩阵每一行的和是否等于1，P.sum()输出一个长度为（P.shape[1],1）的矩阵
    assert (P >= 0.0).all() #判断P中是否所有的元素都大于0，判断P是否是一个有效的概率转移
    new_y = y.copy()
    flipper = np.random.RandomState(random_state) #flipper是一个随机数生成器
    for idx in np.arange(y.shape[0]): #y是一个numpy数组，y.shape[0]用来得出numpy数组
        i = y[idx]
        flipped = flipper.multinomial(1, P[i,:], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    
    return new_y



def noisify_multiclass_symmetric(y_train, noise, random_seed, nb_classes):
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = P * (n/(nb_classes-1))
    if n > 0.0:
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. -n
        P[nb_classes-1, nb_classes-1] = 1. -n

        train_noise_label = multiclass_noisify(y_train, P=P, random_state=random_seed)
        # print("train_noise_label:",train_noise_label)   #[4 8 4 2 7 2 4 2 5 7 8]
        # print("y_train:",y_train)   # [4 4 4 4 8 8 0 2 8 7 8 5 5 ]
        index_noise = train_noise_label != y_train
        actual_noise_rate = (train_noise_label != y_train).mean()
        assert actual_noise_rate >= 0.0

    return train_noise_label, actual_noise_rate, index_noise



def noisify_pairflip(y_train, noise, random_seed, nb_classes):
    """mistakes:
        flip in the pair 类标签只会被翻转到一个非常相似的错误类别
    """
    P = np.eye(nb_classes)
    n = noise
    if(n>0):
        P[0,0], P[0,1] = 1.-n, n
        for i in range(1, nb_classes):
            P[i], P[i+1] = 1.-n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes, 0] = 1.-n, n

        train_noise_label = multiclass_noisify(y_train, P=P, random_state=random_seed)
        index_noise = train_noise_label != y_train
        actual_noise_rate = (train_noise_label != y_train).mean()
        assert actual_noise_rate > 0.0
    
    return train_noise_label, actual_noise_rate, index_noise



def add_noise(args, y_train, user_groups):
    np.random.seed(args.seed)
    #返回一个长度为num_users的numpy数组，每个元素都是0或1
    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_c_initial * gamma_s #得到了每个client对应的噪声比例
    y_train_noisy = copy.deepcopy(y_train) 
    noise_client_list = []

    real_noise_level = np.zeros(args.num_users)
    sample_noise_idx = np.array([])
    for i in np.where(gamma_c > 0)[0]:  #i是某个有噪音的client的id
        sample_idx = np.array(list(user_groups[i]))
        # print("client的sample_idx:",sample_idx) #[34816 37891 49051  8197  5126 44039]
        #把这部分样本以概率gamma_c[i]的概率转化为其他样本
        if args.noise_type == 'pairflip':
            y_train_noisy[sample_idx], actual_noise_rate, noise_index = noisify_pairflip(y_train_noisy[sample_idx], gamma_c[i], args.seed, args.num_classes) 
        elif args.noise_type == 'symmetric':
            y_train_noisy[sample_idx], actual_noise_rate, noise_index = noisify_multiclass_symmetric(y_train_noisy[sample_idx], gamma_c[i], args.seed, args.num_classes)
        print("client %d, noise_level: %.4f, real noise ratio: %.4f" % (i, gamma_c[i], actual_noise_rate))
        real_noise_level[i] = actual_noise_rate
        noise_client_list.append(i)
        sample_noise_idx = np.concatenate([sample_noise_idx, sample_idx[noise_index]])
    print("噪声客户端列表：",noise_client_list)

    return (y_train_noisy, noise_client_list, real_noise_level, sample_noise_idx)
    

       
   


if __name__ == '__main__':
    # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,),
    #                                                         (0.3081,))
    #                                ]))
    # num = 100
    # d = mnist_noniid(dataset_train, num)

    data_dir = '../data/cifar10' #在父附录中(..)找到data/cifar
    trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
    dataset_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_train)

    args = args_parser()
    y_train = np.array(dataset_train.targets)
    num_classes = 10
    non_iid_prob_class = 0.7
    num_users = 50
    alpha_dirichlet = 1
    user_groups = non_iid_dirichlet_sampling(y_train, num_classes, non_iid_prob_class, num_users, args.seed, alpha_dirichlet)
    # print("user_groups:",user_groups) #每个client对应的用户坐标
    #依次输出每个client对应多少个样本类别，每个样本列表用多少个这个列表的样本
    # print("")
