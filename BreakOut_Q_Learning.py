import gym
import numpy as np
from random import randint
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import torch.optim as optim
from copy import deepcopy
import multiprocessing
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Conv2d(1, 5, kernel_size=10, stride=1, padding=1)
        self.l2 = torch.nn.MaxPool2d(kernel_size=5, stride = 3, padding=0)
        self.l3 = torch.nn.Linear(1575, 32)
        self.l4 = torch.nn.Linear(32, 4)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = x.view(-1, 1575)
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
    
class data():
    def __init__(self):
        self.pastactions = list()
        self.reward = list()
        self.states = list()
        self.pastactions_train = list()
        self.qvalues = list()
    def newaction(self, action, reward, state):
        self.states.append(state)
        self.pastactions.append(action)
        self.reward.append(reward)
    def trainrandomdata(self):
        i = randint(0,len(self.states)-2)
        Q = Q_Net(torch.from_numpy(np.array(self.states[i])).float())[0]
        Q_Target = Target_Net(torch.from_numpy(np.array(self.states[i])).float())[0]
        action = self.pastactions[i]
        MaxQIndex = list(Q_Net(torch.from_numpy(np.array(self.states[i + 1])).float())[0].float().detach())
        MaxQIndex = MaxQIndex.index(max(MaxQIndex))
        Q_Target[action] = 0.99 * Target_Net(torch.from_numpy(np.array(self.states[i+1])).float())[0][MaxQIndex] 
        Q_Target[action] = Q_Target[action] + self.reward[i + 1]
        return ((Q-Q_Target)[action])**2
        #print(loss)

def preprocess(x):
    processed_x = list()
    x = x[30:-17]
    a = list()
    for k in range(len(x)):
        if k %3 == 0:
            i = x[k]
            for l in range(len(i[8:-8])):
                if l%2 == 0:
                    j = i[8:-8][l]
                    a.append(sum(j))
            processed_x.append(a)
            a = list()
    return np.array([[processed_x]])
   
#Defines bunch of functions
    def int2list(l):
        a = list()
        for i in range(l):
            a.append(0)
        a.append(1)
        for i in range(3 - l):
            a.append(0)    
        return a

def int2list2(l, k):
    a = list()
    for i in range(l):
        a.append(0)
    a.append(k)
    for i in range(2 - l):
        a.append(0)    
    return a

def maxlist(l1, l2):
    ret = list()
    #print(l2)
    #print(l1)
    for i in range(len(l1)):
        ret.append([list(l1)[i][list(l2[i]).index(max(list(l2[i])))]])
    return ret

def getrandomloss(t):
    return data_list[randint(0,len(data_list)-1)].trainrandomdata()

     
#Sets up memory and Neural Net
data_list = list()
Q_Net = Net()
Target_Net = Net()
#Defines Variables
num_lives = 5
epsilon = 30

#Optimizer and scheduler
optimizer = optim.Adam(Q_Net.parameters(), lr=.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)


for k in range(3000):
    data_list.append(data())
    env = gym.make('Breakout-v0')
    env.reset()
    for i in range(1000):
        
        #Chooses Action
        if i > 1:
            q_vals = list(Q_Net(torch.from_numpy(np.array(processed_state)).float()))
            if randint(0,100) > epsilon:
                action = q_vals.index(max(q_vals))
            else:
                action = randint(0,3)
        else:
            action = 0
        #Excutes Action
        state, r, done, info = env.step(action)
        env.render()
    
        #Defines Reward
        if(info['ale.lives'] < num_lives):
            r = r - 5
            num_lives = info['ale.lives']
        if(info['ale.lives'] == 0):
            break
        #Processes and adds data to memory; Shape is 55x72
        if i > 0:
            processed_state = (preprocess(state) * 2 - preprocess(s0))
            data_list[-1].newaction(action, r, processed_state)  
        s0 = state
        if i > 0 and i%10==0:
            loss = 0
            for k in range(32):
                loss = loss + getrandomloss(1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
    #Updates Target Net
    if k%2 == 0:
        Target_Net = deepcopy(Q_Net)
    env.close()
    scheduler.step()
    epsilon = epsilon * 0.999

    
env.close()
