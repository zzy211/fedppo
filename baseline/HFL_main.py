#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 此文件用来复现hfl (cvpr 2022) 这里主要实现 对称交叉熵+client重加权

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from options import args_parser
from HFL_server.HFL_update import LocalUpdate, test_inference
from HFL_server.HFL_nlp_update import nlpLocalUpdate, nlp_test_inference
from models import ResNet10Cifar, ResNet50Cifar, ResNet18Cifar, CNNEmnist, ResNet10Emnist, LSTM, TextClassificationModel
from utils import get_dataset, average_weights, exp_details
from sampling import add_noise




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


    else:
        exit('Error: unrecognized model')

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
    if not os.path.exists("HFL_acc_record_{}".format(args.dataset)):
        os.mkdir("HFL_acc_record_{}".format(args.dataset))   
    filename = 'HFL_acc_record_{}/HFL_noise_{}_lowerb_{}_frac_{}_user_{}_p_{}_alpha_{}.txt'.format(args.dataset, args.level_n_system, args.level_n_lowerb, args.frac, args.num_users, args.non_iid_prob_class, args.alpha_dirichlet)
    with open(filename,'w') as f:
        f.write('Here is the record of the training of HFL!\n')
    
    client_mean_loss_list = [0 for _ in range(args.num_users)] #存放所有client对应的损失
    amount_with_quality = [0 for _ in range(args.num_users)]


    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_dataset_len = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("选出的client id:", idxs_users) # [96 10 57 97 89 66 72 62 49  5]  [89 99 45 96 72  9 74 88  1 38]
        print("选出来的client集合是:",idxs_users)
        #保存一下当前轮未修改的last_mean_loss_list
        last_mean_loss_list = copy.deepcopy(client_mean_loss_list)
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
            client_mean_loss_list[idx] = copy.deepcopy(loss)
            local_dataset_len.append(local_model.dataset_len)
        
        #计算每一个client的权重
        flag_loss_weight = True
        quality_list = np.zeros(args.num_users)
        quality_sum = 0
        for participant_index in idxs_users:
            deta_loss = last_mean_loss_list[participant_index] - client_mean_loss_list[participant_index]
            if(deta_loss <= 0): #之前没有选择过这个client,此时不通过client的loss进行重加权
                flag_loss_weight = False
                break
            else:
                quality_list[participant_index] = deta_loss/client_mean_loss_list[participant_index]
                quality_sum += quality_list[participant_index]

        # aggregate global weights
        if(flag_loss_weight == True):
            weight_participant = []
            for participant_index in idxs_users:
                amount_with_quality[participant_index] += 0.5 * quality_list[participant_index]/quality_sum
                weight_participant.append(np.exp(amount_with_quality[participant_index]))
            print("!!!!!!!weight_participant:",weight_participant)
            global_weights = average_weights(local_weights, weight_participant) #这里权重不是看local_dataset_len而是看weight_participant
        else:
            global_weights = average_weights(local_weights, local_dataset_len)
       

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        

        # print global training loss after every 'i' rounds
        if args.dataset == 'imdb':
            test_acc, test_loss = nlp_test_inference(args, global_model, x_test=x_test, y_test=y_test)
        else:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
       
        # print(f' \n Results after {args.epochs} global rounds of training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
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
