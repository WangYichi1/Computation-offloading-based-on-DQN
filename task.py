import numpy as np
import math
import constants as cn

class Task():
    def __init__(self, data_size):
        self.data_size = data_size
        self.computation_consumption = self.data_size * cn.cycle_number      
    
    
