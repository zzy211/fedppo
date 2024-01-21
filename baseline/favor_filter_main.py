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
from models import ResNet10Cifar, ResNet50Cifar, ResNet18Cifar, ResNet10Emnist, CNNEmnist
from utils import get_dataset, average_weights, exp_details
from sampling import add_noise
import gym
from ddqn_server.favor_remove_update import *
from ddqn_server.utils import create_directory
from ddqn_server.DDQN import DDQN
from ddqn_server.favor_env import *
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN

class noise_cluster():
    def __init__(self, local_weights_ini, client_acc_list, batch_loss, train_dataset, user_groups):  #(local_weights_ini, client_acc_list, batch_loss, train_dataset, user_groups)
        args = args_parser()
        self.model = args.model
        self.num_users = args.num_users
        self.client_weights = local_weights_ini
        self.client_acc = client_acc_list
        self.batch_loss = batch_loss
        self.train_dataset = train_dataset
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
        pca = PCA(n_components=2)
        compressed_w = pca.fit_transform(w_array_stack)
        
        #-----------clustering-----------------
        if(args.cluster_method == 'hier'):
            Z = linkage(compressed_w, method='ward')
            labels = fcluster(Z, t=2, criterion='distance')
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
            print("noise client:{}, noise rate：{}".format(client_id,noise_rate))
            if(noise_rate > 0.4):
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
    train_dataset, test_dataset, user_groups = get_dataset(args)

    #-------Add Noise to each client-----------#
    y_train = np.array(train_dataset.targets)   
    y_train_noisy, noise_client_list, real_noise_level, sample_noise_id = add_noise(args, y_train, user_groups)  #gamma_s是带噪的client列表
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

    else:
        exit('Error: unrecognized model')


    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    #-----------------------------pre training----------------------------------------#
    local_ini_dir_path = os.path.join(os.getcwd(), 'favor_acc_record/ini')
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
        batch_loss = np.ones(int(len(train_dataset)))
        for client_id in range(args.num_users):
            global_model.train()
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[client_id], batch_loss=batch_loss, global_round=0, client_id=client_id, sample_noise_id = sample_noise_id, important_sample=False, low_noise_client_id=np.array([]), batch_deta_loss=None) #对所有样本进行一次训练
            l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model), global_round=0)
            local_weights_ini.append(copy.deepcopy(w))
            accuracy_id_client, loss_id_client = test_inference(args, l_model, test_dataset) #根据当前local_model和全局的数据集测试一个精度
            client_acc_list.append(accuracy_id_client)
            local_datatset_len_ini.append(local_model.dataset_len)
            for key, value in batch_loss_update.items():
                    batch_loss[key] = value
            
        global_weights = average_weights(local_weights_ini, local_datatset_len_ini)
        # update global weights 
        global_model.load_state_dict(global_weights)
        # Calculate avg training accuracy over all users at every epoch
        global_model.eval()
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        global_client_acc = copy.deepcopy(client_acc_list)
        global_client_acc.insert(0, test_acc)
        # Save the initialized weight model and accuracy
        np.savez(file_ini_path, weights=np.array(local_weights_ini), acc=np.array(global_client_acc))
    
    Ncluster = noise_cluster(local_weights_ini, client_acc_list, batch_loss, train_dataset, user_groups) #local_weights_ini, client_acc_list
    all_maybe_moise_client_id, noise_client_id_find = Ncluster.client_noise_result()   
    if(len(noise_client_id_find)!=0):
        print("!!!!!noise_client_id_find:{}, 查准率：{}, 查全率: {}".format(noise_client_id_find, len(np.intersect1d(noise_client_id_find, noise_client_list))/len(noise_client_id_find), len(np.intersect1d(noise_client_id_find, noise_client_list))/len(noise_client_list)))
    low_noise_client_id = np.setdiff1d(all_maybe_moise_client_id, noise_client_id_find)


    #-------------------------------Create ppo environment---------------------------------------- 
    clean_client_id = [i for i in range(args.num_users) if i not in noise_client_id_find] 
    env = favor_env(clean_client_id)
    ckpt_dir_path = './ddqn_server/checkpoints/DDQN/'
    agent = DDQN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                 fc1_dim=32, fc2_dim=32, ckpt_dir=ckpt_dir_path, gamma=0.95, tau=0.005, epsilon=0.9,
                 eps_end=0.1, eps_dec=0.005, max_size=10000, batch_size=32)  #把batch_size和神经网络调小,把gamma改成了0.5
    create_directory(ckpt_dir_path, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, epsilon_history = [], [], []

    #------------------------------------- Training-----------------------------------------------
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    if not os.path.exists("favor_acc_record"):
        os.mkdir("favor_acc_record") 

    for episode in range(args.max_episodes): #在每个episode里面进行epoch轮联邦平均
        filename = 'favor_acc_record/ddqn_acc_episode_{}_noise_{}_lowerb_{}_frac_{}_user_{}.txt'.format(episode, args.level_n_system, args.level_n_lowerb, args.frac, args.num_users)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('Here is the record of the training acc of fedddqn!\n')
        #初始化网络
        if args.dataset == 'cifar10':
            if args.model == 'resnet50':
                global_model = ResNet50Cifar(args=args)
            elif args.model == 'resnet18':
                global_model = ResNet18Cifar(args=args)
            elif args.model == 'resnet10':
                global_model = ResNet10Cifar(args=args)
        
        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)
        global_weights = global_model.state_dict()
        total_reward = 0    
       
        #把场景中所有client都训练一遍，用所有client的模型权重作为初始状态
        # client_w = {}    #定义一个字典存储所有client的权重
        # for client_id in range(args.num_users):
        #     global_model.train()
        #     local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[client_id])
        #     l_model, w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=0)
        #     client_w[client_id] = copy.deepcopy(w)

        client_w = {}
        for client_id in range(args.num_users):
            client_w[client_id] = copy.deepcopy(local_weights_ini[client_id])

        observation = env.reset(client_w, global_weights) #刚开始时认为所有client的权重都和global_weights一样       

        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses, local_dataset_len = [], [], []
            epoch_reward = 0
            
            print(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1) #favor在训练阶段只选出一个，在测试阶段输出topk个action
            #方案2 一次选出m个client
            all_clients = agent.choose_n_action(observation, k=m, isTrain=True)
            actions_return = all_clients.tolist()
            real_action_list = [clean_client_id[i] for i in actions_return]

            print("all_clients_list:", real_action_list)
            print("噪声client_list:",noise_client_list)
            print("选出了噪声client:",set(real_action_list).intersection(set(noise_client_list)))
            print("当前的状态：",observation)
          
            client_acc_update = {}  #建立一个字典存储更新了的client精度
            for idx in real_action_list:
                # local_model = LocalUpdate(args=args, dataset=train_dataset,
                #                       idxs=user_groups[idx])
                # l_model, w, loss = local_model.update_weights(
                # model=copy.deepcopy(global_model), global_round=epoch)
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], batch_loss=batch_loss, global_round=epoch, client_id=client_id, sample_noise_id = sample_noise_id, important_sample=False, low_noise_client_id=np.array([]), batch_deta_loss=None) #对所有样本进行一次训练
                l_model, w, loss, batch_loss_update = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            
                local_weights.append(copy.deepcopy(w))
                client_w[idx] = copy.deepcopy(w)               
                local_losses.append(copy.deepcopy(loss)) #loss是所有样本的一个平均loss
                local_dataset_len.append(local_model.dataset_len)
                accuracy_id_client, loss_id_client = test_inference(args, l_model, test_dataset) #根据当前local_model和全局的数据集测试一个精度
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
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            global_w = copy.deepcopy(global_weights)
            #这里把返回值从reward_list改为reward了
            observation_, reward_list, done, info = env.step(real_action_list, global_w, client_w, test_acc, args.target_acc, client_acc_update)
            #需要一次存储每一个action对应的s,a,r,s_,认为所有的client在r中具有相同的贡献，都得到100%的r
            #只有第一个客户端被训练,因为之前论文中实际训练了200轮，所以我们这里用2个客户端来模拟来更好的学习到知识
            # first_client_id = all_clients_list[0]
            # agent.remember(observation, first_client_id, reward_list[first_client_id], observation_, done)
            # second_client_id = all_clients_list[1]
            # agent.remember(observation, second_client_id, reward_list[second_client_id], observation_, done)
            # third_client_id = all_clients_list[2]
            # agent.remember(observation, second_client_id, reward_list[third_client_id], observation_,done)

            agent.learn()
            for index, idx in enumerate(real_action_list[:5]):
                action = actions_return[index]
                agent.remember(observation, action, reward_list[idx], observation_, done)
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
        test_acc, test_loss = test_inference(args, global_model, test_dataset) #全局模型在全局测试数据集上的测试精度

        print(f' \n Results after {episode} episode global rounds of training:')
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        print("|---- Test loss: {:.2f}".format(test_loss))
        print("|---- total_reward: {:.2f}%".format(total_reward))
 

        #每一个回合结束的时候要保存一下模型
        if(episode + 1) % 5 == 0:
            agent.save_models(episode+1)


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


