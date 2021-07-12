from env import Maze
from dqn import DQN
from constants import *
import pdb
import warnings
import torch
import numpy as np
import operator
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from random import choice
from TD3 import TD3
from utils import ReplayBuffer
from Q_table import QLearningTable
from func import trans
ACTIONS = []
for i in range(66):
    ACTIONS.append(i)
warnings.filterwarnings("ignore")

r = []
rl_r = []
e = []
Time = []
rl_Time = []
action = torch.Tensor([])
RL = QLearningTable(ACTIONS)

epsilon = 0.9995
MEMORY_CAPACITY = 1000
#pdb.set_trace()
env = Maze(local_f)

agent = DQN()
sum_ = 0
rl_reward_sum = 0
t_sum = 0
rl_sum = 0

for episode in range(30000):
       
    observation = env.reset()
    action,ep = agent.choose_action(observation)
    rl_action,rl_flag,rl_ep = RL.choose_action(str(observation))
    ep = round(ep,2)
    
    t_action = action

    if t_action == 65:
        t_action = [0,2,1]
    elif t_action == 64:
        t_action = [0,2,0]
    elif t_action == 63:
        t_action = [0.1,2,1]
    elif t_action == 62:
        t_action = [0.1,2,0]
    elif t_action == 61:
        t_action = [0.2,2,1]
    elif t_action == 60:
        t_action = [0.2,2,0]
    elif t_action == 59:
        t_action = [0.3,2,1]
    elif t_action == 58:
        t_action = [0.3,2,0]
    elif t_action == 57:
        t_action = [0.4,2,1]
    elif t_action == 56:
        t_action = [0.4,2,0]
    elif t_action == 55:
        t_action = [0.5,2,1]
    elif t_action == 54:
        t_action = [0.5,2,0]
    elif t_action == 53:
        t_action = [0.6,2,1]
    elif t_action == 52:
        t_action = [0.6,2,0]
    elif t_action == 51:
        t_action = [0.7,2,1]
    elif t_action == 50:
        t_action = [0.7,2,0]
    elif t_action == 49:
        t_action = [0.8,2,1]
    elif t_action == 48:
        t_action = [0.8,2,0]
    elif t_action == 47:
        t_action = [0.9,2,1]
    elif t_action == 46:
        t_action = [0.9,2,0]
    elif t_action == 45:
        t_action = [1,2,1]
    elif t_action == 44:
        t_action = [1,2,0]
        
    elif t_action == 43:
        t_action = [0,1,1]
    elif t_action == 42:
        t_action = [0,1,0]
    elif t_action == 41:
        t_action = [0,0,1]
    elif t_action == 40:
        t_action = [0,0,0]
    elif t_action == 39:
        t_action = [0.1,1,1]
    elif t_action == 38:
        t_action = [0.1,1,0]
    elif t_action == 37:
        t_action = [0.1,0,1]
    elif t_action == 36:
        t_action = [0.1,0,0]
    elif t_action == 35:
        t_action = [0.2,1,1]
    elif t_action == 34:
        t_action = [0.2,1,0]
    elif t_action == 33:
        t_action = [0.2,0,1]
    elif t_action == 32:
        t_action = [0.2,0,0]
    elif t_action == 31:
        t_action = [0.3,1,1]
    elif t_action == 30:
        t_action = [0.3,1,0]
    elif t_action == 29:
        t_action = [0.3,0,1]
    elif t_action == 28:
        t_action = [0.3,0,0]
    elif t_action == 27:
        t_action = [0.4,1,1]
    elif t_action == 26:
        t_action = [0.4,1,0]
    elif t_action == 25:
        t_action = [0.4,0,1]
    elif t_action == 24:
        t_action = [0.4,0,0]
    elif t_action == 23:
        t_action = [0.5,1,1]
    elif t_action == 22:
        t_action = [0.5,1,0]
    elif t_action == 21:
        t_action = [0.5,0,1]
    elif t_action == 20:
        t_action = [0.5,0,0]
    elif t_action == 19:
        t_action = [0.6,1,1]
    elif t_action == 18:
        t_action = [0.6,1,0]
    elif t_action == 17:
        t_action = [0.6,0,1]
    elif t_action == 16:
        t_action = [0.6,0,0]
    elif t_action == 15:
        t_action = [0.7,1,1]
    elif t_action == 14:
        t_action = [0.7,1,0]
    elif t_action == 13:
        t_action = [0.7,0,1]
    elif t_action == 12:
        t_action = [0.7,0,0]
    elif t_action == 11:
        t_action = [0.8,1,1]
    elif t_action == 10:
        t_action = [0.8,1,0]
    elif t_action == 9:
        t_action = [0.8,0,1]
    elif t_action == 8:
        t_action = [0.8,0,0]
    elif t_action == 7:
        t_action = [0.9,1,1]
    elif t_action == 6:
        t_action = [0.9,1,0]
    elif t_action == 5:
        t_action = [0.9,0,1]
    elif t_action == 4:
        t_action = [0.9,0,0]
    elif t_action == 3:
        t_action = [1,1,1]
    elif t_action == 2:
        t_action = [1,1,0]
    elif t_action == 1:
        t_action = [1,0,1]
    else:
        t_action = [1,0,0]
    
    observation_, reward ,t= env.step(t_action,action)
    rl_t_action = trans(rl_action)
    u,rl_reward,rl_t = env.step(rl_t_action,rl_action)
    #if  and
    if episode != 0:
        #print(episode,rl_t)
        sum_ += reward
        rl_reward_sum += rl_reward
        re = sum_/episode
        rl_re = rl_reward_sum/episode
        r.append(re)
        rl_r.append(rl_re)
        rl_sum += rl_t
        #print("总的",rl_sum)
        t_sum += t
        tt = t_sum/episode
        rl_tt = rl_sum/episode
        #print("平均",rl_tt)
        e.append(episode)
        Time.append(tt)
        rl_Time.append(rl_tt)
    
    #file_handle=open('ran.txt',mode='a')
    #if agent.flag is 1 and agent.memory_counter >= MEMORY_CAPACITY:
    #file_handle.write("epsilon"+ str(rl_ep)+"       state"+str(observation)+"      action:"+str(rl_action)+"      reward:"+str(rl_reward)+"      t"+str(rl_t)+"\n")
    #if agent.memory_counter < MEMORY_CAPACITY:
     #   print("action:",t_action,"reward:",reward)
    agent.store_transition(observation, action, reward, observation_)
    RL.learn(str(observation), rl_action, rl_reward, str(observation_))

    observation = observation_
    if agent.memory_counter >= MEMORY_CAPACITY:
        agent.learn()
#file_handle.close()

plt.figure()
plt.plot(e[10:],Time[10:],'b', label='DQN')
plt.plot(e[10:],rl_Time[10:],'r', label='Q-learning')
plt.legend()
plt.title("时延（舍弃10个点）")
plt.show()

plt.figure()
plt.plot(e[10:],r[10:],'b', label='DQN')
plt.plot(e[10:],rl_r[10:],'r', label='Q-learning')
plt.legend()
plt.title("reward（舍弃10个点）")
plt.show()

plt.figure()
plt.plot(e,Time,'b', label='DQN')
plt.plot(e,rl_Time,'r', label='Q-learning')
plt.legend()
plt.title("时延")
plt.show()

plt.figure()
plt.plot(e,r,'b', label='DQN')
plt.plot(e,rl_r,'r', label='Q-learning')
plt.legend()
plt.title("reward")
plt.show()
