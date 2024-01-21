#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# -----fedppo-------

import os
import copy
import time
import pickle
from typing import Any
import numpy as np
from tqdm import tqdm

import torch

from options import args_parser
from models import ResNet10Cifar, ResNet50Cifar, ResNet18Cifar, CNNEmnist, ResNet10Emnist, LSTM, TextClassificationModel
from utils import get_dataset, average_weights, exp_details
from sampling import add_noise
import gym
from ppo_server.ppo import PPO
from ppo_server.ppo_remove_update import LocalUpdate, test_inference, nlpLocalUpdate, nlp_test_inference
from gym import logger, spaces
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
np.set_printoptions(threshold=np.inf)

class fed_env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, clean_client_id):     
        args = args_parser()
        self.num_users = len(clean_client_id)
        self.action_space = spaces.Discrete(self.num_users)   #action space is all clients
        low = np.zeros(self.num_users+1)  
        high = np.ones(self.num_users+1) 
        self.observation_space = spaces.Box(low, high, shape=(self.num_users+1,), dtype=np.float32)       
        self.target_acc = args.target_acc
        self.state = np.ones(self.num_users + 1)
      
    def step(self, action_n, client_acc_update, global_acc, client_deta_loss_update):  
        self.state[0] = global_acc
        reward_list = []
        for index, client_id in enumerate(action_n):
            self.state[client_id + 1] = client_acc_update[index]
            alpha  = 1
            beta = 1
            r = alpha * client_acc_update[index] + beta * global_acc * client_deta_loss_update[index]/sum(client_deta_loss_update)
            reward_list.append(r)
        return np.array(self.state), reward_list, done, {}

    def reset(self):
        self.N_a = np.zeros(self.num_users)
        self.state = np.zeros(self.num_users + 1)
        return np.array(self.state)


class noise_cluster():
    def __init__(self, local_weights_ini, client_acc_list, batch_loss, user_groups):  #(local_weights_ini, client_acc_list, batch_loss, train_dataset, user_groups)
        args = args_parser()
        self.model = args.model
        self.num_users = args.num_users
        self.client_weights = local_weights_ini
        self.client_acc = client_acc_list
        self.batch_loss = batch_loss
        self.user_groups = user_groups
       
    def client_update(self, id, weight, acc):
        self.client_weights[id] = weight
        self.client_acc[id] = acc

    def client_noise_result(self):
        w_array = []    #Used to store the flattened weights of all clients
        for l_w in self.client_weights:
            w_k = np.array([])
            for key, weights in l_w.items():
                if 'resnet' in self.model:
                    if 'conv' in key:
                        weights_cpu = weights.cpu().detach().numpy()
                        weights_cpu_flatten = weights_cpu.flatten()
                        w_k = np.concatenate([w_k, weights_cpu_flatten])
                else:
                    weights_cpu = weights.cpu().detach().numpy()
                    weights_cpu_flatten = weights_cpu.flatten()
                    w_k = np.concatenate([w_k, weights_cpu_flatten])
            w_k = w_k.reshape(1, -1)[0]
            w_array.append(w_k)
        w_array_stack = np.stack(w_array, axis=0)
        pca = PCA(n_components=10)
        compressed_w = pca.fit_transform(w_array_stack)
        
        #-----------clustering-----------------
        if(args.cluster_method == 'hier'):
            Z = linkage(compressed_w, method='ward')
            labels = fcluster(Z, t=2.5, criterion='distance')
            labels -= 1
            cluster_num = len(set(labels)) - (1 if -1 in labels else 0)
        elif(args.cluster_method == 'gmm'):
            gmm = GaussianMixture(n_components=2)
            gmm.fit(compressed_w)
            labels = gmm.predict(compressed_w)
            cluster_num = 2
        elif(args.cluster_method == 'dbscan'):
            dbscan = DBSCAN(eps=0.6, min_samples=2)
            dbscan.fit(compressed_w)
            labels = dbscan.labels_
            cluster_num = len(set(labels)) - (1 if -1 in labels else 0)
        elif(args.cluster_method == 'kmeans'):
            cluster_num = 2
            kmeans = KMeans(n_clusters=cluster_num)
            kmeans.fit(compressed_w)
            labels = kmeans.labels_
        else:   #不进行预训练
            return {}, []

        cluster_dict = {}
        cluster_acc = []
        for cluster_id in range(0, cluster_num):
            client_index = np.where(labels == cluster_id)[0]
            cluster_dict[cluster_id] = client_index
            avg_acc = 0
            for i in client_index:
                avg_acc += self.client_acc[i]
            avg_acc = avg_acc/len(client_index)
            cluster_acc.append(avg_acc)
        min_cluster_id = cluster_acc.index(min(cluster_acc))
        noise_client_id = []
        for client_id in cluster_dict[min_cluster_id]:  
            idxs_sample = self.user_groups[client_id]
            batch_loss_now = np.array([self.batch_loss[i] for i in idxs_sample])
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(batch_loss_now.reshape(-1, 1))
            labels = gmm.predict(batch_loss_now.reshape(-1, 1))
            means = gmm.means_
            nums = [len(labels)-sum(labels), sum(labels)]
            max_mean_id = 0 if means[0] > means[1] else 1
            noise_rate = nums[max_mean_id] / len(labels)
            print("noise client:{}, noise rate：{}, acc:{}".format(client_id,noise_rate,self.client_acc[client_id]))
            if(noise_rate > 0.5):
                noise_client_id.append(client_id)                
        return cluster_dict[min_cluster_id], noise_client_id
            


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

    #----------- BUILD MODEL--------------------#
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


    #-----------------------------pre training----------------------------------------#
    local_ini_dir_path = os.path.join(os.getcwd(), 'fedppo_acc_record_{}/ini'.format(args.dataset))
    if not os.path.exists(local_ini_dir_path):
        os.makedirs(local_ini_dir_path)
    file_ini_name = 'local_ini_2_{}_{}_{}.npz'.format(args.num_users, args.level_n_system, args.level_n_lowerb)
    file_ini_path = os.path.join(local_ini_dir_path, file_ini_name)
    if os.path.exists(file_ini_path):
        print("Pre-trained weights and precision already exist, load them!")
        local_ini_data = np.load(file_ini_path, allow_pickle=True)
        local_weights_ini = local_ini_data['weights']
        global_client_acc = local_ini_data['acc']
        batch_loss = local_ini_data['loss']
        client_acc_list = global_client_acc[1:]
    else:
        local_weights_ini = []
        client_acc_list = []
        local_datatset_len_ini = []
        if args.dataset == "imdb":
            batch_loss = np.ones(int(len(x_train)))
        else:
            batch_loss = np.ones(int(len(train_dataset)))
        for client_id in range(args.num_users):
            global_model.train()
            if args.dataset == 'imdb':
                local_model = nlpLocalUpdate(args=args, x_train=x_train, y_train=y_train, idxs=user_groups[client_id])
                l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model),global_round=0)
                accuracy_id_client, loss_id_client = nlp_test_inference(args, l_model, x_test=x_test, y_test=y_test)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[client_id], batch_loss=batch_loss, global_round=0, client_id=client_id, sample_noise_id = sample_noise_idx, important_sample=False, low_noise_client_id=np.array([]), batch_deta_loss=None) #对所有样本进行一次训练
                l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model), global_round=0)
                accuracy_id_client, loss_id_client = test_inference(args, l_model, test_dataset)
            client_acc_list.append(accuracy_id_client)
            local_weights_ini.append(copy.deepcopy(w))
            local_datatset_len_ini.append(local_model.dataset_len)
            for key, value in batch_loss_update.items():
                    batch_loss[key] = value
        global_weights = average_weights(local_weights_ini, local_datatset_len_ini)
        # update global weights 
        global_model.load_state_dict(global_weights)
        # Calculate avg training accuracy over all users at every epoch
        global_model.eval()
        if args.dataset == 'imdb':
            test_acc, test_loss = nlp_test_inference(args, global_model, x_test=x_test, y_test=y_test)
        else:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
        global_client_acc = copy.deepcopy(client_acc_list)
        global_client_acc.insert(0, test_acc)
        # Save the initialized weight model and accuracy
        np.savez(file_ini_path, weights=np.array(local_weights_ini), acc=np.array(global_client_acc), loss=batch_loss)
    
    Ncluster = noise_cluster(local_weights_ini, client_acc_list, batch_loss, user_groups) #local_weights_ini, client_acc_list
    all_maybe_moise_client_id, noise_client_id_find = Ncluster.client_noise_result()   
    if(len(noise_client_id_find)!=0 and len(noise_client_list)!=0):
        print("!!!!!noise_client_id_find:{}, 查准率：{}, 查全率: {}".format(noise_client_id_find, len(np.intersect1d(noise_client_id_find, noise_client_list))/len(noise_client_id_find), len(np.intersect1d(noise_client_id_find, noise_client_list))/len(noise_client_list)))
    low_noise_client_id = np.setdiff1d(all_maybe_moise_client_id, noise_client_id_find)

    

    #-------------------------------Create ppo environment----------------------------------------
    clean_client_id = [i for i in range(args.num_users) if i not in noise_client_id_find]
    env = fed_env(clean_client_id)

    ppo_agent = PPO(args=args, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
    lr_actor=0.001, lr_critic=0.001, gamma=0.7, K_epochs=5, eps_clip=0.2, has_continuous_action_space=False)

    total_rewards, avg_rewards, epsilon_history = [], [], []

    #------------------------------------- Training-----------------------------------------------
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    ppo_record_name = 'fedppo_acc_record_{}/ppo_record_noise_{}_lowerb_{}_user_{}_cluster_{}'.format(args.dataset, args.level_n_system, args.level_n_lowerb, args.num_users, args.cluster_method)
    with open(ppo_record_name, 'w') as f:
        f.write('Here is the record of the training of fedppo!\n')


    if args.dataset == 'imdb':
        batch_deta_loss = np.ones(int(len(x_train))) * pow(10, -5)
    else:
        batch_deta_loss = np.ones(int(len(train_dataset))) * pow(10, -5)
    for episode in range(args.max_episodes):
        filename = 'fedppo_acc_record_{}/ppo_acc_episode_{}_noise_{}_lowerb_{}_frac_{}_user_{}_p_{}_alpha_{}_cluster_{}.txt'.format(args.dataset, episode, args.level_n_system, args.level_n_lowerb, args.frac, args.num_users, args.non_iid_prob_class, args.alpha_dirichlet, args.cluster_method)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('Here is the record of the training acc of fedppo!\n')
        
        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)
        # copy weights
        global_weights = global_model.state_dict()
        total_reward = 0    
        done = False 
        observation = env.reset()
        client_mean_loss_list = [0 for _ in range(args.num_users)]  #存放所有client对应的loss
       
        for epoch in tqdm(range(args.epochs)):
            
            local_weights, local_losses, local_dataset_len = [], [], []
            epoch_reward = 0
            
            print(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)            
            action_list, action_logprob_list, state_val_list = ppo_agent.select_n_actions(observation)
            action_return = [action.item() for action in action_list]
            real_actions_list = [clean_client_id[i] for i in action_return] #Map the action output by rl back to the real action
            
            print("all_clients_list:",real_actions_list)
            print("noise client_list:",noise_client_list)
            print("choose noise client:",set(real_actions_list).intersection(set(noise_client_list)))
            with open(ppo_record_name, 'a') as f:
                f.write("\nepoch:{} choose the client:{} choose the noise client:{}".format(epoch, real_actions_list, set(real_actions_list).intersection(set(noise_client_list))))

            client_acc_update = []  
            client_deta_loss_update = []
            last_mean_loss_list = copy.deepcopy(client_mean_loss_list)
            for idx in real_actions_list:
                if args.dataset == 'imdb':
                    local_model = nlpLocalUpdate(args=args, x_train=x_train, y_train=y_train, idxs=user_groups[idx])
                    l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model),global_round=epoch)
                    accuracy_id_client, loss_id_client = nlp_test_inference(args, l_model, x_test=x_test, y_test=y_test)
                else:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx],  batch_loss=batch_loss, global_round=epoch, client_id=idx, sample_noise_id = sample_noise_idx, important_sample=False, low_noise_client_id=low_noise_client_id, batch_deta_loss=batch_deta_loss) #在这里把目前的batch_loss传进去以进行重要性采样
                    l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)                 
                    accuracy_id_client, loss_id_client = test_inference(args, l_model, test_dataset)
                
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss)) 
                client_mean_loss_list[idx] = copy.deepcopy(loss)
                deta_loss = max(last_mean_loss_list[idx] - copy.deepcopy(loss), pow(10,-5))
                client_deta_loss_update.append(deta_loss)
                local_dataset_len.append(local_model.dataset_len)
                client_acc_update.append(accuracy_id_client)
                
                #update batch_loss
                batch_loss_before = batch_loss.copy()
                for key, value in batch_loss_update.items():
                    batch_loss[key] = value
                    if(batch_loss_before[key] - batch_loss[key] > 0):
                        batch_deta_loss[key] = batch_loss_before[key] - batch_loss[key] 
            
                   

            # update global weights
            global_weights = average_weights(local_weights, local_dataset_len)

            # update global weights 
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            
            # Calculate avg training accuracy over all users at every epoch
            global_model.eval()
            if args.dataset == 'imdb':
                test_acc, test_loss = nlp_test_inference(args, global_model, x_test=x_test, y_test=y_test)
            else:
                test_acc, test_loss = test_inference(args, global_model, test_dataset)
            observation_, reward_list, done, info = env.step(action_return, client_acc_update, global_acc=test_acc, client_deta_loss_update=client_deta_loss_update)

            
            for index in range(len(action_return)):
                ppo_agent.remember(observation, action_list[index], action_logprob_list[index], state_val_list[index], reward_list[index], done)
                epoch_reward += reward_list[index]
            if epoch % 5 == 0:
                ppo_agent.update()
          
            
            observation = observation_

            print("epoch:",epoch,"epoch_reward:",epoch_reward)
            total_reward += epoch_reward
            with open(ppo_record_name, 'a') as f:
                f.write("\nepoch:"+str(epoch)+" epoch_reward:"+str(epoch_reward))

                

            # print global training loss after every 'i' rounds          
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print('Test Accuracy: {:.2f}% \n'.format(100*test_acc))
            print('Test loss: {:.2f} \n'.format(test_loss))
            with open(ppo_record_name, 'a') as f:
                f.write("\nepoch:"+str(epoch)+" epoch_acc:"+str(test_acc)+" epoch_loss:"+str(test_loss))
            
            with open(filename, 'a', encoding='utf-8') as f:
                f.write("\nepoch:"+str(epoch)+" epoch_acc:"+str(test_acc)+" epoch_reward:"+str(epoch_reward))
            if(test_acc>=args.target_acc):
                break

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print(f' \n Results after {episode} episode global rounds of training:')
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        print("|---- Test loss: {:.2f}".format(test_loss))
        print("|---- total_reward: {:.2f}%".format(total_reward))
        with open(ppo_record_name, 'a') as f:
            f.write("\nepisode:"+str(episode)+" episode_reward:"+str(total_reward)+" episode_acc:"+str(test_acc)+" episode_loss:"+str(test_loss))
        with open('fedppo_acc_record_{}/ddqn_reward.txt'.format(args.dataset), 'a', encoding='utf-8') as f:
             f.write("\nepisode:"+str(episode)+" episode_acc:"+str(test_acc)+" episode_reward:"+str(total_reward))
        

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


