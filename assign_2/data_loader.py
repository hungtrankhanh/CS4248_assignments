'''
Name : Tran Khanh Hung (A0212253W)
Name : Lim Jia Xian Clarence (A0212209U)
'''
import torch
import collections
import numpy as np

class DataLoader():
    def __init__(self, X,y, batch_size):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        # dequeu to limit buffer and auto remove element when full
        self.X_train = X
        self.y_train = y
        self.batch_size = batch_size
        self.data_size = len(y)




    def sample(self):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''

        # random a batch of samples
        print("-------------data_size = ", self.data_size)
        indices = np.random.choice(self.data_size, self.batch_size, replace=False)
        return self.X_train[indices], self.y_train[indices]


    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)