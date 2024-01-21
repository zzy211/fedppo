#----复现 favor（fedddqn)的代码----#
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# -----fedd3qn-------

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
import gym
from ddqn_server.utils import create_directory
from ddqn_server.DDQN import DDQN
from ddqn_server.D3QN import D3QN 
from ddqn_server.favor_env import *


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

    #--------------Bulid Model--------------#
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

    #Create d3qn environment
    # env = gym.make('CartPole-v0')
    env = favor_env()
    ckpt_dir_path = './ddqn_server/checkpoints/DDQN/'
    if(args.RL_name == 'ddqn'):
        agent = DDQN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                    fc1_dim=32, fc2_dim=32, ckpt_dir=ckpt_dir_path, gamma=0.95, tau=0.005, epsilon=0.9,
                    eps_end=0.1, eps_dec=0.005, max_size=10000, batch_size=32)  #把batch_size和神经网络调小,把gamma改成了0.5
    elif(args.RL_name == 'd3qn'):
        agent = D3QN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                    fc1_dim=32, fc2_dim=32, ckpt_dir=ckpt_dir_path, gamma=0.95, tau=0.005, epsilon=0.9,
                    eps_end=0.1, eps_dec=0.005, max_size=10000, batch_size=32)  #把batch_size和神经网络调小,把gamma改成了0.5

    create_directory(ckpt_dir_path, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, epsilon_history = [], [], []

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    if not os.path.exists("favor_acc_record_{}_{}".format(args.dataset, args.RL_name)):
        os.mkdir("favor_acc_record_{}_{}".format(args.dataset, args.RL_name)) 

    for episode in range(args.max_episodes): #在每个episode里面进行epoch轮联邦平均
        filename = 'favor_acc_record_{}_{}/ddqn_acc_episode_{}_noise_{}_lowerb_{}_frac_{}_user_{}.txt'.format(args.dataset, args.RL_name, episode, args.level_n_system, args.level_n_lowerb, args.frac, args.num_users)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('Here is the record of the training acc of fedddqn!\n')
        
        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)
        global_weights = global_model.state_dict()

        total_reward = 0    
        #把场景中所有client都训练一遍，用所有client的模型权重作为初始状态
        client_w = {}    #定义一个字典存储所有client的权重
        for client_id in range(args.num_users):
            global_model.train()
            if args.dataset == 'imdb':
                local_model = nlpLocalUpdate(args=args, x_train=x_train, y_train=y_train, idxs=user_groups[client_id])
                l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model),global_round=0)
                client_w[client_id] = copy.deepcopy(w) 
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[client_id])
                l_model, w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=0)
                client_w[client_id] = copy.deepcopy(w) 

       
        observation = env.reset(client_w, global_weights) #刚开始时认为所有client的权重都和global_weights一样       

        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses, local_dataset_len = [], [], []
            epoch_reward = 0
            
            print(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1) #favor在训练阶段只选出一个，在测试阶段输出topk个action
            #方案2 一次选出m个client
            all_clients = agent.choose_n_action(observation, k=m, isTrain=True)
            all_clients_list = all_clients.tolist() 
            print("all_clients_list:",all_clients_list)
            print("噪声client_list:",noise_client_list)
            print("选出了噪声client:",set(all_clients_list).intersection(set(noise_client_list)))
            print("当前的状态：",observation)
          
            client_acc_update = {}  #建立一个字典存储更新了的client精度
            for idx in all_clients_list:
                if args.dataset == 'imdb':
                    local_model = nlpLocalUpdate(args=args, x_train=x_train, y_train=y_train, idxs=user_groups[client_id])
                    l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model),global_round=epoch)
                    accuracy_id_client, loss_id_client = nlp_test_inference(args, l_model, x_test=x_test, y_test=y_test) #根据当前local_model和全局的数据集测试一个精度
                    
                else:
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
                    l_model, w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                    accuracy_id_client, loss_id_client = test_inference(args, l_model, test_dataset) #根据当前local_model和全局的数据集测试一个精度
           
                
            
                local_weights.append(copy.deepcopy(w))
                client_w[idx] = copy.deepcopy(w)               
                local_losses.append(copy.deepcopy(loss)) #loss是所有样本的一个平均loss
                local_dataset_len.append(local_model.dataset_len)
                client_acc_update[idx] = accuracy_id_client


            # update global weights 对所选的m个client的模型进行聚合得到全局模型
            global_weights = average_weights(local_weights, local_dataset_len)

            # update global weights 更新全局模型的权重
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            
            # Calculate avg training accuracy over all users at every epoch
            global_model.eval()
            #测试一下当前全局模型的精度
            if args.dataset == 'imdb':
                test_acc, test_loss = nlp_test_inference(args, global_model, x_test=x_test, y_test=y_test)
            else:
                test_acc, test_loss = test_inference(args, global_model, test_dataset)
            global_w = copy.deepcopy(global_weights)
            #这里把返回值从reward_list改为reward了
            observation_, reward_list, done, info = env.step(all_clients_list, global_w, client_w, test_acc, args.target_acc, client_acc_update)
            #需要一次存储每一个action对应的s,a,r,s_,认为所有的client在r中具有相同的贡献，都得到100%的r
            #只有第一个客户端被训练,因为之前论文中实际训练了200轮，所以我们这里用2个客户端来模拟来更好的学习到知识
            # first_client_id = all_clients_list[0]
            # agent.remember(observation, first_client_id, reward_list[first_client_id], observation_, done)
            # second_client_id = all_clients_list[1]
            # agent.remember(observation, second_client_id, reward_list[second_client_id], observation_, done)
            # third_client_id = all_clients_list[2]
            # agent.remember(observation, second_client_id, reward_list[third_client_id], observation_,done)

            agent.learn()
            for idx in all_clients_list[:5]:
                agent.remember(observation, idx, reward_list[idx], observation_, done)
                agent.learn()
                epoch_reward += reward_list[idx]
            # epoch_reward = reward
            observation = observation_



            print("epoch:",epoch,"epoch_reward:",epoch_reward)
            total_reward += epoch_reward
    

                

            # print global training loss after every 'i' rounds          
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print('Test Accuracy: {:.2f}% \n'.format(100*test_acc))
            print('Test loss: {:.2f} \n'.format(test_loss))

            with open(filename, 'a', encoding='utf-8') as f:
                f.write("\nepoch:"+str(epoch)+" epoch_reward:"+str(epoch_reward)+" epoch_loss:"+str(test_loss)+" epoch_acc:"+str(test_acc))  
            
            if(test_acc>=args.target_acc):
                break

        # Test inference after completion of training
        if args.dataset == 'imdb':
            test_acc, test_loss = nlp_test_inference(args, global_model, x_test=x_test, y_test=y_test)
        else:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
        
        print(f' \n Results after {episode} episode global rounds of training:')
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        print("|---- Test loss: {:.2f}".format(test_loss))
        print("|---- total_reward: {:.2f}%".format(total_reward))
 

        #每一个回合结束的时候要保存一下模型
        if(episode + 1) % 5 == 0:
            agent.save_models(episode+1)


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


