#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import math



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        index = self.idxs[item]
        return torch.tensor(image), torch.tensor(label), index

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=100):
        super(SCELoss, self).__init__()  
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
    
    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels.long())
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels.to(torch.int64), self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-7, max=1.0)
        rce = (-1*torch.sum(pred*torch.log(label_one_hot), dim=1))
        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs,  batch_loss, global_round, client_id, sample_noise_id, important_sample, low_noise_client_id, batch_deta_loss): #idxs是client有的数据的id
        self.args = args
        # self.logger = logger
        self.batch_loss = batch_loss
        self.sample_noise_id = sample_noise_id
        self.trainloader, self.dataset_len = self.train_val_test(
            dataset, list(idxs), global_round, client_id, important_sample, low_noise_client_id, batch_deta_loss)
        self.device = 'cuda' if args.gpu_id else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss(reduction='none').to(self.device)
        self.criterion = SCELoss(alpha=0.6, beta=0.4, num_classes=args.num_classes)

       
        
   

    def train_val_test(self, dataset, idxs, global_round, client_id, important_sample, low_noise_client_id, batch_deta_loss):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        
        idxs_train = idxs
      
    


        if(len(idxs_train) > 500 and important_sample): 
            batch_deta_loss_now = np.array([batch_deta_loss[i] for i in idxs_train])
            weights = batch_deta_loss_now / np.sum(batch_deta_loss_now)
            sample1 = np.random.choice(idxs_train, size=int(len(idxs_train)*0.3), replace=True, p=weights)
            sample2 = np.random.choice(idxs_train, size=int(len(idxs_train)*0.7), replace=False)   
            idxs_train_sample = np.concatenate([sample1, sample2])
            
        else:
            idxs_train_sample = idxs_train

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train_sample),
                                 batch_size=self.args.local_bs, shuffle=True)  
      
        return trainloader, len(idxs_train_sample)


    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        batch_loss_update = {}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, indexs) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                per_sample_loss = self.criterion(log_probs, labels.long())
                loss = per_sample_loss.mean()
                # print("per_sample_loss:",per_sample_loss)
        
                for i in range(len(per_sample_loss)):
                    batch_loss_update[indexs[i]] = per_sample_loss[i].item()
                
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model, model.state_dict(), sum(epoch_loss) / len(epoch_loss), batch_loss_update

                
         

        

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels.long())
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu_id else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


class nlpLocalUpdate(object):
    def __init__(self, args, x_train, y_train, idxs):
        self.args = args
        self.device = 'cuda' if args.gpu_id else 'cpu'
        self.criterion = nn.NLLLoss(reduction='none').to(self.device)
        self.user_idxs = idxs
        self.train_loader, self.shuffle_index_list = self.get_train_loader(x_train=x_train ,y_train=y_train)
        self.dataset_len = len(idxs)

      
    def get_train_loader(self, x_train, y_train):
        x_train_selected = [x_train[i] for i in self.user_idxs]
        y_train_selected = [y_train[i] for i in self.user_idxs]
        train_data = TensorDataset(torch.LongTensor(x_train_selected), torch.LongTensor(y_train_selected))

        train_sampler = RandomSampler(train_data) #会在每个epoch中对数据随机进行采样
        shuffled_indices = iter(train_sampler)
        shuffle_index_list = []
        real_index_list = list(self.user_idxs)
        for _ in range(0, len(train_data)):
            index = next(shuffled_indices)
            real_index = real_index_list[index]
            shuffle_index_list.append(real_index)

        
        # print("indexs:",self.user_idxs," shuffle_index_list:", shuffle_index_list)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.local_bs)
        return train_loader, shuffle_index_list

    
    def update_weights(self, model, global_round):
        model.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        batch_loss_update = {}
        epoch_loss = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (datas, labels) in enumerate(self.train_loader):
                datas, labels = datas.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(datas)
                # loss =  criterion(log_probs, labels)
                if(log_probs.dim() == 1):
                    log_probs = torch.unsqueeze(log_probs, dim=0)
                # labels = torch.unsqueeze(labels, dim=0)
                
                per_sample_loss = criterion(log_probs, labels)
                for i in range(0, len(per_sample_loss)):
                    sample_num = batch_idx*self.args.local_bs + i
                    sample_index = self.shuffle_index_list[sample_num]
                    batch_loss_update[sample_index] = per_sample_loss[i]
   
                loss = per_sample_loss.mean()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(datas),
                            len(self.train_loader.dataset),
                            100. * batch_idx / len(self.train_loader), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return model, model.state_dict(), sum(epoch_loss) / len(epoch_loss), batch_loss_update
            




def nlp_test_inference(args, model, x_test, y_test):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, acc = 0.0, 0.0, 0.0
    device = 'cuda' if args.gpu_id else 'cpu'
    criterion = nn.NLLLoss().to(device)
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    test_sampler = SequentialSampler(test_data) #顺序采样
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=128)
    for batch_idx, (datas, labels) in enumerate(test_loader):
        datas, labels = datas.to(device), labels.to(device)
        with torch.no_grad():
            log_probs = model(datas)
        batch_loss = criterion(log_probs, labels)
        loss += batch_loss.item()
        pred_labels = log_probs.max(-1, keepdim=True)[1]  
        acc += pred_labels.eq(labels.view_as(pred_labels)).sum().item() 
    loss /= len(test_loader.dataset)
    acc /= len(test_loader.dataset)
    return acc, loss
