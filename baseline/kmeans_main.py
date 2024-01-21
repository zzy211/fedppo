#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from options import args_parser
from update import LocalUpdate, test_inference
from ppo_server.ppo_nlp_update import nlpLocalUpdate, nlp_test_inference
from models import ResNet10Cifar, ResNet50Cifar, ResNet18Cifar, CNNEmnist, ResNet10Emnist, LSTM, TextClassificationModel
from utils import get_dataset, average_weights, exp_details
from sampling import add_noise
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    exp_details(args)

    if args.gpu_id:
        torch.cuda.set_device(int(args.gpu_id))
    device = 'cuda' if args.gpu_id else 'cpu'

    # load dataset and user groups
    if args.dataset == 'imdb':
        x_train, y_train, x_test, y_test, user_groups = get_dataset(args)
    else:
        train_dataset, test_dataset, user_groups = get_dataset(args)

    #-------Add Noise to each client-----------#
    if args.dataset == 'imdb':
        target_train = np.array(y_train)
        y_train_noisy, noise_client_list, real_noise_level, sample_noise_idx = add_noise(args, target_train, user_groups)  #gamma_s是带噪的client列表
        y_train = y_train_noisy
    else:
        y_train = np.array(train_dataset.targets)  
        y_train_noisy, noise_client_list, real_noise_level, sample_noise_idx = add_noise(args, y_train, user_groups)  #gamma_s是带噪的client列表
        train_dataset.targets = y_train_noisy

     # BUILD MODEL
    if args.dataset == 'clothing1m':
        if args.model == 'resnet50':
            global_model = ResNet50Cifar(args=args)
        elif args.model == 'resnet18':
            global_model = ResNet18Cifar(args=args)
        elif args.model == 'resnet10':
            global_model = ResNet10Cifar(args=args)

    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.model == 'resnet50':
            global_model = ResNet50Cifar(args=args)
        elif args.model == 'resnet18':
            global_model = ResNet18Cifar(args=args)
        elif args.model == 'resnet10':
            global_model = ResNet10Cifar(args=args)
    
    elif args.dataset == 'emnist':
        if args.model == 'resnet10':
            global_model = ResNet10Emnist(args=args)
        elif args.model == 'cnn':
            global_model = CNNEmnist(args=args)
    
    elif args.dataset == 'imdb':
        if args.model == 'lstm':
            global_model = LSTM(args=args)
        elif args.model == 'dnn':
            global_model = TextClassificationModel(args=args)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    if not os.path.exists('kmeans_acc_record_{}'.format(args.dataset)):
        os.mkdir('kmeans_acc_record_{}'.format(args.dataset))
    filename = 'kmeans_acc_record_{}/kmeans_noise_{}_lowerb_{}_frac_{}_user_{}.txt'.format(args.dataset, args.level_n_system, args.level_n_lowerb, args.frac, args.num_users)
    with open(filename,'w') as f:
        f.write('Here is the record of the training of kmeans!\n')
    
    #对场景中所有的client训练一遍，用kmeans进行聚类
    local_weights_ini = [] #用来记录每个client训练一遍的结果
    for client_id in range(args.num_users):
        global_model.train()
        if args.dataset == 'imdb':
            local_model = nlpLocalUpdate(args=args, x_train=x_train, y_train=y_train, idxs=user_groups[client_id])
            l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model),global_round=0)
        else:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[client_id])
            l_model, w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=0)
        local_weights_ini.append(copy.deepcopy(w))
    #得到所有client的权重后进行一次聚类
    w_array = []  #用来存储所有client被展平后的权重
    #1.先将所有client的权重展平，再pca压缩
    for l_w_ini in local_weights_ini:
        #当前client的权重是l_w_ini, 用w_k来保存当前client的展平后的权重
        w_k = np.array([])  
        for key, weights in l_w_ini.items():
            if(('resnet' in args.model and 'conv' in key) or 'resnet' not in args.model):
                weights_cpu = weights.cpu().detach().numpy()
                weights_cpu_flatten = weights_cpu.flatten()
                w_k = np.concatenate([w_k, weights_cpu_flatten])
        w_k = w_k.reshape(1, -1)[0]
        w_array.append(w_k)
    w_array_stack = np.stack(w_array, axis=0)
    
    pca = PCA(n_components=args.num_users)  #压缩成2维看一下
    pca2 = PCA(n_components=2)
    compressed_w = pca.fit_transform(w_array_stack) #维度是num_users * num_users
    compressed_w_2= pca2.fit_transform(w_array_stack)
    #保存一下客户端的权重

    # cluster_num = max(int(args.frac * args.num_users), 1)
    cluster_num = 15
    kmeans = KMeans(n_clusters= cluster_num)
    for c in [compressed_w, compressed_w_2]:
        kmeans.fit(c)
        #输出每个client聚类后的结果
        labels = kmeans.labels_
        cluster_dict = {}
        print("cluster_num:",cluster_num)
        for cluster_id in range(0, cluster_num):
            print("cluster_id:",cluster_id)
            client_index = np.where(labels == cluster_id)[0]
            cluster_dict[cluster_id] = client_index
            print("cluster_dict[cluster_id]:",cluster_dict[cluster_id])
    #选出来每个簇里的一个client来参与每次训练
    
# '''
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_dataset_len = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = []
        for cluster_id in range(0, m):
            random_id = np.random.choice(cluster_dict[cluster_id])
            idxs_users.append(random_id)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("选出的client id:", idxs_users) # [96 10 57 97 89 66 72 62 49  5]  [89 99 45 96 72  9 74 88  1 38]
        print("选出来的client集合是:",idxs_users)

        for idx in idxs_users:
            #用全局模型和client i的本地数据集进行更新 传client i的样本user_groups[idx]
            if args.dataset == 'imdb':
                local_model = nlpLocalUpdate(args=args, x_train=x_train, y_train=y_train, idxs=user_groups[idx])
                l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model),global_round=epoch)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
                l_model, w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss)) #loss是所有样本的一个平均loss
            local_dataset_len.append(local_model.dataset_len)


        # update global weights
        global_weights = average_weights(local_weights, local_dataset_len)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
       

        if args.dataset == 'imdb':
            test_acc, test_loss = nlp_test_inference(args, global_model, x_test=x_test, y_test=y_test)
        else:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)

        with open(filename,'a') as f:
            f.write('\n epoch:'+str(epoch)+" acc:"+str(test_acc))
        if(test_acc>=args.target_acc):
            break

        
        

    
    

    

    
    

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
# '''