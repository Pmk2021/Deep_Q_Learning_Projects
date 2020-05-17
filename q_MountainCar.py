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
import multiprocessing as mp
from sklearn.preprocessing import Normalizer

loss = 0

pool = mp.Pool(mp.cpu_count())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(4,50)
        self.l3 = nn.Linear(50,2)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l3(x)
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
        i = randint(0,len(self.states)-1)
        if i == len(self.states)-1:
            Q = Q_Net(torch.from_numpy(np.array(self.states[i])).float())
            action = self.pastactions[i]
            return (Q[action] - self.reward[i])**2
        Q = Q_Net(torch.from_numpy(np.array(self.states[i])).float())
        #Q_Target = Target_Net(torch.from_numpy(np.array(self.states[i])).float())
        action = self.pastactions[i]
        MaxQIndex = list(Q_Net(torch.from_numpy(np.array(self.states[i + 1])).float()).float().detach())
        MaxQIndex = MaxQIndex.index(max(MaxQIndex))
        Q_Target = 0.85 * Target_Net(torch.from_numpy(np.array(self.states[i+1])).float())[MaxQIndex] + self.reward[i]
        return ((Q[action]-Q_Target))**2
        #print(loss)
    def trainall(self):
        loss0 = 0
        for i in range(len(self.pastactions) - 1):
            Q = Q_Net(torch.from_numpy(np.array(self.states[i])).float())
            Q_Target = Target_Net(torch.from_numpy(np.array(self.states[i])).float())
            action = self.pastactions[i]
            MaxQIndex = list(Q_Net(torch.from_numpy(np.array(self.states[i + 1])).float()).float().detach())
            MaxQIndex = MaxQIndex.index(max(MaxQIndex))
            Q_Target[action] = 0.99 * Target_Net(torch.from_numpy(np.array(self.states[i+1])).float())[MaxQIndex] + self.reward[i + 1]
        
        #optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        #optimizer.step()
            loss0 = loss0 + sum((Q - Q_Target)**2)
        return loss0
    def trainrandbatch(self, size):
        n = randint(0, len(self.reward) - 2)
        if n + size > (len(self.reward)):
            actionbatch = self.pastactions[n:]
            statebatch = self.states[n:]
            rewardbatch = self.reward[n:]
        else:
            actionbatch = self.pastactions[n: n + size]
            statebatch = self.states[n: n + size]
            rewardbatch = self.reward[n: n + size]
            
  
        action0 = list()
        maxqval = Target_Net(torch.from_numpy(np.array(statebatch[1:])).float()).detach().numpy()
        maxqval = torch.from_numpy(np.array(maxlist(maxqval)))
        for i in actionbatch:
            action0.append(int2list(i))
        reward0 = np.array(rewardbatch[1:]).reshape(len(rewardbatch[1:]), 1)
        loss1 = loss + sum(sum(((torch.from_numpy(reward0) * -1 + c * maxqval * -1 +
                          Q_Net(torch.from_numpy(np.array(statebatch[0:-1])).float()))**2) * torch.from_numpy(np.array(action0[:-1])).float()))
        return loss1
        #optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        #optimizer.step()
        #print(loss)
        ''' def returnrandlist(self):
        n = randint(0,len(self.reward)-2)
        naction = self.pastactions[n]
        nstate = self.states[n]
        nreward = self.reward[n+1]'''
    def getlen(self):
        return len(self.states)
    def trainlast(self):
        i = -2
        action = self.pastactions[i]
        Q = Q_Net(torch.from_numpy(np.array(self.states[i])).float())[action]
        MaxQIndex = list(Q_Net(torch.from_numpy(np.array(self.states[i + 1])).float()).float().detach())
        MaxQIndex = MaxQIndex.index(max(MaxQIndex))
        Q_Target  = 0.85 * Target_Net(torch.from_numpy(np.array(self.states[i+1])).float())[MaxQIndex] + self.reward[i + 1]
        return ((Q-Q_Target))**2

        
Q_Net = Net()
Target_Net = Net()

#Q_Net.load_state_dict(torch.load(r'C:\Users\pmuralikrishnan\Desktop\Python\New folder\New folder\MountainCar_Q_Policy'))
#Q_Net.eval()

#Target_Net.load_state_dict(torch.load(r'C:\Users\pmuralikrishnan\Desktop\Python\New folder\New folder\MountainCar_Q_Policy'))
#Target_Net.eval()

Q_Data = data()

optimizer = optim.Adam(Q_Net.parameters(), lr=.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

c = 0.1

def int2list(l):
    a = list()
    for i in range(l):
        a.append(0)
    a.append(1)
    for i in range(2 - l):
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
    return datalist[randint(0,len(datalist)-1)].trainrandomdata()

prob1 = 30
prob2 = 100
epochs = 100
ep_len = 3000
datalist = list()
complete = False
norm_list1 = list()
norm_list2 = list()
max_point = -0.5

a = 200
av_len = 0
for j in range(300000):
    env = gym.make('CartPole-v1')
    s1 = env.reset()
    datalist.append(data())
    complete = False
    far_dist = -100
    fin = False
    for i in range(a):
        q_vals = list(Q_Net(torch.from_numpy(np.array(s1)).float()))
        if randint(0,100) > prob1:
            action = q_vals.index(max(q_vals))
        else:
            action = randint(0,1)
        state, r, done, info = env.step(action)
        if j%10000000 == 0 and 1 == 3:
            env.render()
        s0 = s1
        s1 = state
        datalist[-1].newaction(action, r, s0)
        if done:
            r = -5
        if i > 2 and i%2 == 0:
            loss = 0
            for k in range(1):
                loss = loss + getrandomloss(1)
                #loss = sum([pool.apply(getrandomloss, args=(row)) for row in range(16)])
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
        if done:
            av_len += i
            fin = True
            break
    env.close()
    if not fin:
        print('win')
        scheduler.step()
        prob1 = prob1 * 0.95
        av_len += a
    #print(loss)
    if j%100 == 0:
        Target_Net = deepcopy(Q_Net)
    a = 200
    prob1 = prob1 * 0.999
    if j%100 == 0:
        print(av_len/100)
        av_len = 0
    #scheduler.step()
    #prob = prob * 0.99
#torch.save(Q_Net.state_dict(), r'C:\Users\pmuralikrishnan\Desktop\Python\New folder\New folder\MountainCar_Q_Policy')
