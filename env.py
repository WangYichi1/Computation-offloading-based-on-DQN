import numpy as np
from math import *
from my_random import ran, ran01, result
from constants import *

class Maze(object):
    def __init__(self, local_f):
        self.local_f = local_f
        self.task = None
        self.MEC1 = None
        self.MEC2 = None
        self.MECS = None
        self.Prtask = Prtask
        self.user_X = user_X
        self.user_Y = user_Y
        self.K = None
        self.i = 0
        self.last_task = None
        self.last_MEC1 = None
        self.last_MEC2 = None
        self.last_MEC3 = None
        
    def reset(self):
        self.last_task = self.task
        self.K = self.set_channel()
        self.MEC1 = self.random_MEC1()
        self.MEC2 = self.random_MEC2()
        self.MEC3 = self.random_MEC3()
        self.MECS = [self.MEC1,self.MEC2,self.MEC3]
        self.task = self.random_create_task()
        observation = self.get_state()
        return observation

    def set_channel(self):
        p1 = np.random.random()
        if self.K is None:
            k1 = 0.1
        else:
            '''
            if self.K[0] == 0.1:
                if 0 <= p1 < 0.01:
                    k1 = 0.1
                elif 0.01 <= p1 < 0.99:
                    k1 = 0.2
                else:
                    k1 = 0.9
            elif self.K[0] == 0.2:
                if 0 <= p1 < 0.01:
                    k1 = 0.1
                elif 0.01 <= p1 < 0.02:
                    k1 = 0.2
                else:
                    k1 = 0.9
            else:
                if 0 < p1 <= 0.98:
                    k1 = 0.1
                elif 0.98 < p1 <= 0.99:
                    k1 = 0.2
                else:
                    k1 = 0.9
                    '''
            if self.K[0] == 0.1:
                if 0 <= p1 < 1/3:
                    k1 = 0.1
                elif 1/3 <= p1 < 2/3:
                    k1 = 0.2
                else:
                    k1 = 0.9
            elif self.K[0] == 0.2:
                if 0 <= p1 < 1/3:
                    k1 = 0.1
                elif 1/3 <= p1 < 2/3:
                    k1 = 0.2
                else:
                    k1 = 0.9
            else:
                if 0 < p1 <= 1/3:
                    k1 = 0.1
                elif 1/3 < p1 <= 2/3:
                    k1 = 0.2
                else:
                    k1 = 0.9
                    
        p2 = np.random.random()
        if self.K is None:
            k2 = 0.6
        else:
            '''
            if self.K[1] == 0.6:
                if 0 <= p2 < 0.01:
                    k2 = 0.6
                elif 0.01 <= p2 < 0.99:
                    k2 = 0.4
                else:
                    k2 = 0.01
            elif self.K[1] == 0.4:
                if 0 <= p2 < 0.01:
                    k2 = 0.6
                elif 0.01 <= p2 < 0.02:
                    k2 = 0.4
                else:
                    k2 = 0.01
            else:
                if 0 < p2 <= 0.98:
                    k2 = 0.6
                elif 0.98 < p2 <= 0.99:
                    k2 = 0.4
                else:
                    k2 = 0.01
                    '''
            if self.K[1] == 0.6:
                if 0 <= p2 < 1/3:
                    k2 = 0.6
                elif 1/3 <= p2 < 2/3:
                    k2 = 0.4
                else:
                    k2 = 0.01
            elif self.K[1] == 0.4:
                if 0 <= p2 < 1/3:
                    k2 = 0.6
                elif 1/3 <= p2 < 2/3:
                    k2 = 0.4
                else:
                    k2 = 0.01
            else:
                if 0 < p2 <= 1/3:
                    k2 = 0.6
                elif 1/3 < p2 <= 2/3:
                    k2 = 0.4
                else:
                    k2 = 0.01
        K = np.array([k1,k2])
        return K

    def random_create_task(self):
        p = np.random.random()
        if self.last_task is None:
            data = data_size[0]
        else:
            if self.last_task.data_size == data_size[0]:
                if 1/4 > p >= 0:
                    data = data_size[0]
                elif 1/2 > p >= 1/4:
                    data = data_size[1]
                elif 3/4 > p >= 1/2:
                    data = data_size[2]
                else:
                    data = data_size[3]
            elif self.last_task.data_size == data_size[1]:
                if 1/4 > p >= 0:
                    data = data_size[0]
                elif 1/2 > p >= 1/4:
                    data = data_size[1]
                elif 3/4 > p >= 1/2:
                    data = data_size[2]
                else:
                    data = data_size[3]
            elif self.last_task.data_size == data_size[2]:
                if 1/4 > p >= 0:
                    data = data_size[0]
                elif 1/2 > p >= 1/4:
                    data = data_size[1]
                elif 3/4 > p >= 1/2:
                    data = data_size[2]
                else:
                    data = data_size[3]
            else:
                if 1/4 > p >= 0:
                    data = data_size[0]
                elif 1/2 > p >= 1/4:
                    data = data_size[1]
                elif 3/4 > p >= 1/2:
                    data = data_size[2]
                else:
                    data = data_size[3]
        return Task(data)

    def random_MEC1(self):
        #MEC_fre_1 = np.random.normal(12*GHZ,1*GHZ)
        p1 = np.random.random()
        if self.last_MEC1 is None:
            MEC_fre_1 = 10*GHZ
        else:
            '''
            if self.last_MEC1 == 10*GHZ:
                if 0 <= p1 < 0.01:
                    MEC_fre_1 = 10*GHZ
                elif 0.01 <= p1 < 0.99:
                    MEC_fre_1 = 0.2*GHZ
                else:
                    MEC_fre_1 = 104*GHZ
            elif self.last_MEC1 == 0.2*GHZ:
                if 0 <= p1 < 0.01:
                    MEC_fre_1 = 10*GHZ
                elif 0.01 <= p1 < 0.02:
                    MEC_fre_1 = 0.2*GHZ
                else:
                    MEC_fre_1 = 104*GHZ
            else:
                if 0 < p1 <= 0.98:
                    MEC_fre_1 = 10*GHZ
                elif 0.98 < p1 <= 0.99:
                    MEC_fre_1 = 0.2*GHZ
                else:
                    MEC_fre_1 = 104*GHZ
                    '''
            if self.last_MEC1 == 10*GHZ:
                if 0 <= p1 < 1/3:
                    MEC_fre_1 = 10*GHZ
                elif 1/3 <= p1 < 2/3:
                    MEC_fre_1 = 0.2*GHZ
                else:
                    MEC_fre_1 = 104*GHZ
            elif self.last_MEC1 == 0.2*GHZ:
                if 0 <= p1 < 1/3:
                    MEC_fre_1 = 10*GHZ
                elif 1/3 <= p1 < 2/3:
                    MEC_fre_1 = 0.2*GHZ
                else:
                    MEC_fre_1 = 104*GHZ
            else:
                if 0 < p1 <= 1/3:
                    MEC_fre_1 = 10*GHZ
                elif 1/3 < p1 <= 2/3:
                    MEC_fre_1 = 0.2*GHZ
                else:
                    MEC_fre_1 = 104*GHZ
        self.last_MEC1 = MEC_fre_1
        return MEC_fre_1
    
    def random_MEC2(self):
        #MEC_fre_2 = np.random.normal(10*GHZ,1*GHZ)
        p2 = np.random.random()
        if self.last_MEC2 is None:
            MEC_fre_2 = 4*GHZ
        else:
            '''
            if self.last_MEC2 == 0.1*GHZ:
                if 0 <= p2 < 0.01:
                    MEC_fre_2 = 0.1*GHZ
                elif 0.01 <= p2 < 0.99:
                    MEC_fre_2 = 20*GHZ
                else:
                    MEC_fre_2 = 3*GHZ
            elif self.last_MEC2 == 20*GHZ:
                if 0 <= p2 < 0.01:
                    MEC_fre_2 = 0.1*GHZ
                elif 0.01 <= p2 < 0.02:
                    MEC_fre_2 = 20*GHZ
                else:
                    MEC_fre_2 = 3*GHZ
            else:
                if 0 < p2 < 0.98:
                    MEC_fre_2 = 0.1*GHZ
                elif 0.98 <= p2 < 0.99:
                    MEC_fre_2 = 20*GHZ
                else:
                    MEC_fre_2 = 3*GHZ
                    '''
            if self.last_MEC2 == 4*GHZ:
                if 0 <= p2 < 1/3:
                    MEC_fre_2 = 4*GHZ
                elif 1/3 <= p2 < 2/3:
                    MEC_fre_2 = 20*GHZ
                else:
                    MEC_fre_2 = 0.01*GHZ
            elif self.last_MEC2 == 20*GHZ:
                if 0 <= p2 < 1/3:
                    MEC_fre_2 = 4*GHZ
                elif 1/3 <= p2 < 2/3:
                    MEC_fre_2 = 20*GHZ
                else:
                    MEC_fre_2 = 0.01*GHZ
            else:
                if 0 < p2 < 1/3:
                    MEC_fre_2 = 4*GHZ
                elif 1/3 <= p2 < 2/3:
                    MEC_fre_2 = 20*GHZ
                else:
                    MEC_fre_2 = 0.01*GHZ
        self.last_MEC2 = MEC_fre_2
        return MEC_fre_2

    def random_MEC3(self):
        #MEC_fre_2 = np.random.normal(10*GHZ,1*GHZ)
        p3 = np.random.random()
        if self.last_MEC3 is None:
            MEC_fre_3 = 40*GHZ
        else:
            if self.last_MEC3 == 40*GHZ:
                if 0 <= p3 < 1/3:
                    MEC_fre_3 = 40*GHZ
                elif 1/3 <= p3 < 2/3:
                    MEC_fre_3 = 7*GHZ
                else:
                    MEC_fre_3 = 0.2*GHZ
            elif self.last_MEC3 == 7*GHZ:
                if 0 <= p3 < 1/3:
                    MEC_fre_3 = 40*GHZ
                elif 1/3 <= p3 < 2/3:
                    MEC_fre_3 = 7*GHZ
                else:
                    MEC_fre_3 = 0.2*GHZ
            else:
                if 0 < p3 < 1/3:
                    MEC_fre_3 = 40*GHZ
                elif 1/3 <= p3 < 2/3:
                    MEC_fre_3 = 7*GHZ
                else:
                    MEC_fre_3 = 0.2*GHZ
            self.last_MEC3 = MEC_fre_3
        return MEC_fre_3
    
    def get_state(self):
        #k1 = round(self.K[0],2)
        #k2 = round(self.K[1],1)
        m = np.array([self.K[0],self.K[1]])
        x = np.concatenate(([self.task.data_size/(10*KB)],[self.MEC1/(10*GHZ)],[self.MEC2/(10*GHZ)],[self.MEC3/(10*GHZ)],m.reshape(-1)))
        x = np.array(x)
        return x

    def step(self, action,a):
        #print(action)
        action = np.array(action)
        t_local = (1 - action[0]) * self.task.computation_consumption / self.local_f
        distance = sqrt(pow(self.user_X - MECS_x[int(action[1])],2)+pow(self.user_Y - MECS_y[int(action[1])],2))
        x = P_send * self.K[int(action[2])]* pow(1/distance,2)/noise
        rate = width * log((1 + x), 2)
        t_send = action[0]* self.task.data_size / rate + action[0]* self.task.computation_consumption / self.MECS[int(action[1])]
        #t_send = (1 - action[0]) * self.task.data_size / self.rate[0] + (1 - action[0]) * self.task.computation_consumption / self.MECS[0]

        if t_local > t_send :
            t = t_local
        else:
            t = t_send
            
        reward = 0
        
        #print(a,t)
        
        if t > slot:
            reward = -1
        else:
            reward = exp(-10*t)
            #reward = 13.55*reward-12.26
        
        #print(t)
        '''
        if a == 3:
            r = 1
        elif a == 1 or a == 2:
            r = 0
        else:
            r = -1
        '''
        observation_ = self.reset()
        return observation_, reward,t
     
class Task():
    def __init__(self, data_size):
        self.data_size = data_size
        self.computation_consumption = self.data_size * cycle_number      
    
