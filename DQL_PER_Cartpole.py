import gym
import numpy as np
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

loss = 0

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
        self.states2 = list()
        self.qvalues = list()
        self.isdone = list()
        self.pos = 0
    def newaction(self, action, reward, state, next_state, done):
        #if len(self.states) > 2000:
        #    for j in range(len(state)):
        #        state[j] = (state[j] - self.mean_list[j])/(self.stdev_list[j] + .01)
        #    for j in range(len(state)):
        #        next_state[j] = (next_state[j] - self.mean_list[j])/(self.stdev_list[j] + 0.01)
        if len(self.states) == 4096:
            self.pos = self.pos%4096
            self.states[self.pos] = state
            self.states2[self.pos] = next_state
            self.pastactions[self.pos] = action
            self.isdone[self.pos] = done
            self.reward[self.pos] = reward
            Data_Tree.new_data(reward**2 + 0.1,self.pos)
            self.pos = (self.pos + 1)%4096
        else:
            self.states.append(state)
            self.isdone.append(done)
            self.states2.append(next_state)
            self.pastactions.append(action)
            self.reward.append(reward)
            Data_Tree.new_data(0.5,self.pos)
            self.pos += 1
    def normalize_data(self, state):
        if len(self.states) > 2000:
            for j in range(len(state)):
                state[j] = (state[j] - self.mean_list[j])/(self.stdev_list[j] + .000001)
        return state
            
    def trainrandomdata(self):
        i, pri = Data_Tree.random_sample()
        pri = float(pri)
        if self.isdone[i]:
            Q = Q_Net(torch.from_numpy(np.array(self.states[i])).float())
            action = self.pastactions[i]
            Q_loss = (Q[action] - self.reward[i])**2
            Data_Tree.update_priority(i,Q_loss)
            total = float(Data_Tree.total_pri())
            return Q_loss * (total/pri)**b
        Q = Q_Net(torch.from_numpy(np.array(self.states[i])).float())
        #Q_Target = Target_Net(torch.from_numpy(np.array(self.states[i])).float())
        action = self.pastactions[i]
        MaxQIndex = list(Q_Net(torch.from_numpy(np.array(self.states2[i])).float()).float().detach())
        MaxQIndex = MaxQIndex.index(max(MaxQIndex))
        Q_Target = 0.99 * Target_Net(torch.from_numpy(np.array(self.states2[i])).float())[MaxQIndex] + self.reward[i]
        #Q_loss = huber_loss(Q[action],Q_Target)
        Q_loss = (Q[action]-Q_Target)**2
        Data_Tree.update_priority(i, Q_loss)
        total = float(Data_Tree.total_pri())
        return Q_loss * (total/pri)**b
    
    def calc_norm(self):
        states_copy = np.array(self.states)
        self.mean_list = list()
        self.stdev_list = list()
        for i in range(len(states_copy[0])):
            self.mean_list.append(np.mean(states_copy[:,i]))
            self.stdev_list.append(np.std(states_copy[:,i]))
 
          
    
class tree():
    def __init__(self):
        self.lists = [[{'pri':0.0, 'num':-1} for j in range(2**i)] for i in range(13)]
        self.num_filled = 0
        self.len = 2**12
    def new_data(self, priority, number):
        self.num_filled = number
        priority_change = priority - float(self.lists[-1][number]['pri'])
        self.lists[-1][self.num_filled] = {'pri':priority, 'num':number}
        current_pos = int((self.num_filled)/2)
        for i in reversed(range(len(self.lists)-1)):
            self.lists[i][current_pos]['pri'] += priority_change
            current_pos = int(current_pos/2)
        self.num_filled += 1
        self.num_filled = self.num_filled % self.len
    def random_sample(self):
        cur_pos = 0
        for i in self.lists[1:-1]:
            n = randint(0,99)   
            if n < i[cur_pos]['pri']/(i[cur_pos]['pri'] + i[cur_pos + 1]['pri']) * 100:
               cur_pos = cur_pos * 2
            else:
                 cur_pos = (cur_pos+1) * 2 
        n = randint(0,99)
        if n < self.lists[-1][cur_pos]['pri']/(self.lists[-1][cur_pos]['pri'] + self.lists[-1][cur_pos+1]['pri']) * 100:
            return self.lists[-1][cur_pos]['num'], self.lists[-1][cur_pos]['pri']
        return self.lists[-1][cur_pos+1]['num'], self.lists[-1][cur_pos+1]['pri']
    def update_priority(self, num, new_priority):
        priority_change = new_priority - float(self.lists[-1][num]['pri'])
        cur_pos = num
        for i in reversed(range(len(self.lists))):
            self.lists[i][cur_pos]['pri'] += priority_change
            cur_pos = int(cur_pos/2)
    def total_pri(self):
        total_priority = float(self.lists[0][0]['pri'])
        return total_priority

def huber_loss(d1, d2):
    l = d1 - d2
    if torch.abs(l) < 1:
        return 1/2 * l**2
    else:
        return torch.abs(l) - 1/2            
        
def getrandomloss(t):
   return Q_Data.trainrandomdata()  
      
Q_Net = Net()
#Q_Net.load_state_dict(torch.load(r'C:\Users\pmuralikrishnan\Desktop\Python\New folder\New folder\Cartpole_Q_PER_Policy'))
#Q_Net.eval()

Target_Net = deepcopy(Q_Net)

Q_Data = data()

Data_Tree = tree()

optimizer = optim.Adam(Q_Net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
epsilon = 20
epochs = 10000
b = 0.4
batch_size = 1000
rew = 0

a = 200
for epoch in range(epochs):
    env = gym.make('CartPole-v0')# 4 states 2 actions
    s1 = env.reset()
    #if epoch > 2:
    #   s1 = Q_Data.normalize_data(s1)
    num_died = 0
    if epoch%100 == 0:
        print(str(rew/100))
        rew = 0
    for time_step in range(batch_size):
        q_vals = list(Q_Net(torch.from_numpy(np.array(s1)).float()))
        if randint(0,100) > epsilon:
            action = q_vals.index(max(q_vals))
        else:
            action = randint(0,1)
        state, r, done, info = env.step(action)
        #if epoch > 2:
        #    state = Q_Data.normalize_data(state)
        rew += r
        if  epoch%20 == 0 and 0 == 0:
            env.render()
        s0 = s1
        s1 = state
        Q_Data.newaction(action, r, s0, s1,done)
        if done:
            num_died += 1
        if (num_died > 2 or epoch > 2) and time_step%2 == 0:
            loss = 0
            for k in range(16):
                loss = loss + getrandomloss(1)
                #loss = sum([pool.apply(getrandomloss, args=(row)) for row in range(16)])
            #nn.utils.clip_grad_norm_(Q_Net.parameters(), 1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        if done:
            env.close()
            break
            s1 = env.reset()
    Q_Data.calc_norm()
    env.close()
    scheduler.step()
    if epsilon > 20:
        epsilon = epsilon - 1
    if epoch%10 == 0:
        Target_Net = deepcopy(Q_Net)
    b += 0.0005
    if b > 1:
        b = 1
    
    a = 200
    #print('Reward:' +str(rew) + 'Epoch:' + str(epoch))
torch.save(Q_Net.state_dict(), r'C:\Users\pmuralikrishnan\Desktop\Python\New folder\New folder\Cartpole_Q_PER_Policy')