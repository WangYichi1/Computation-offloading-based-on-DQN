import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.1
EPISILO = 0.995
MEMORY_CAPACITY = 1000
Q_NETWORK_ITERATION = 100

NUM_ACTIONS = 66
NUM_STATES = 6

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.flag = 0
        self.epsilon = 0.9999

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= self.epsilon:# greedy policy
            action_value = self.eval_net.forward(state)
            #if self.memory_counter > MEMORY_CAPACITY:
             #   print(action_value)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] #if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            self.flag = 1
            self.epsilon = self.epsilon * 0.9999
            #print(self.epsilon)
            #if self.memory_counter > MEMORY_CAPACITY:
             #   print(action)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            self.flag = 0
            #action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action,self.epsilon


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        #print("1",self.eval_net(batch_state))
        #print("2",batch_action)
        #print("q_eval",q_eval)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        #print("batch_reward",batch_reward)
        #print("GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)",GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1))
        #print("q_target",q_target)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #for parms in self.eval_net.parameters():
         #   print('-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
'''
def main():
    dqn = DQN()
    episodes = 400
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, _ , done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
        r = copy.copy(reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        

if __name__ == '__main__':
    main()
'''
