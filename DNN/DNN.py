# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:57:48 2020

@author:ly
"""
import torch
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pickle
# import matplotlib.pyplot as plt
# default network
seed=12
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Python random module.
torch.manual_seed(seed) 

device = torch.device("cpu")
torch.set_num_threads(64)

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def R_2(y_pred, y_true):
    return 1 - ((y_pred - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(nn.Linear(2048, 2048), 
                                   nn.ReLU(),
                                    nn.Linear(2048, 1024),
                                    nn.ReLU(), 
                                    # nn.Linear(512, 256),
                                    # nn.ReLU(),
                                    nn.Linear(1024, 1))

    def forward(self, input1):
        output = self.model(input1)
        return output
    def predict(self, input_):
        return self.model(input_)
    
with open('D1A1_ECFP_input.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
    
tasks, train_dataset, val_dataset, test_dataset = data_list
tasks.append('s1-t1')
train_y = train_dataset[:,2048:2057]
train_s1_t1 = train_y[:,0] - train_y[:,6]
train_y = np.hstack((train_y, train_s1_t1.reshape(len(train_s1_t1),1)))

test_y = test_dataset[:,2048:2057]
test_s1_t1 = test_y[:,0] - test_y[:,6]
test_y = np.hstack((test_y, test_s1_t1.reshape(len(test_s1_t1),1)))

val_y = val_dataset[:,2048:2057]
val_s1_t1 = val_y[:,0] - val_y[:,6]
val_y = np.hstack((val_y, val_s1_t1.reshape(len(val_s1_t1),1)))


#training
X_train = train_dataset[:,0:2048]
X_train = torch.Tensor(X_train).to(device)

X_val = val_dataset[:,0:2048]
X_val  = torch.Tensor(X_val).to(device)

y_train_dataset = train_y
y_train_dataset = torch.Tensor(y_train_dataset).to(device)

y_val_dataset = val_y
y_val_dataset = torch.Tensor(y_val_dataset).to(device)


# hyper parameters
LR = 0.001
BATCH_SIZE = 32
EPOCH = list(range(400))

# tasks = tasks[0:2]
all_val_loss=[]
all_train_loss=[]

for i in tasks:
    start_time=time.time()
    val_loss_all=[]
    train_loss_all=[]

    id_=tasks.index(i)
    y_train = y_train_dataset[:,id_]
    y_train = y_train.reshape(y_train_dataset.shape[0],1)
    y_val = y_val_dataset[:,id_]
    y_val = y_val.reshape(y_val_dataset.shape[0],1)
    net_Adam = Net()
    nets = net_Adam
    
    opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
    optimizers = opt_Adam
    
    loss_func = torch.nn.MSELoss()
    
    #initial loss
    train_output=nets(X_train)
    train_loss=loss_func(train_output,y_train)
    val_output=nets(X_val)
    val_loss=loss_func(val_output,y_val)

    train_loss_all.append(train_loss.item())
    val_loss_all.append(val_loss.item())
    for epoch in EPOCH:
        output = nets(X_train) # get output for every net
        loss = loss_func(output,y_train) # compute loss for every net
        optimizers.zero_grad() # claer gradients for net train
        loss.backward() # backpropagation, compute gradients
        optimizers.step() # apply gradients
        # loss_.append(loss.item()) # loss recoder
        if (epoch+1) % 10 == 0:
            train_output=nets(X_train)
            train_loss=loss_func(train_output,y_train)
            val_output=nets(X_val)
            val_loss=loss_func(val_output,y_val)
        
            train_loss_all.append(train_loss.item())
            val_loss_all.append(val_loss.item())
    
    print(i)
    end_time=time.time()
    print(end_time-start_time)
    
    all_val_loss.append(val_loss_all)
    all_train_loss.append(train_loss_all)
    
all_val_loss_array=np.array(all_val_loss).T
all_train_loss_array=np.array(all_train_loss).T
epoch_final = list(range(0,410,10))
for q in range(all_val_loss_array.shape[1]):
    task_id=tasks[q]
    loss_data=all_val_loss_array[:,q].tolist()
    sort_id_ascend=np.argsort(loss_data)[0]
    
    print(task_id,':score:{}/{}'.format(epoch_final[sort_id_ascend],loss_data[sort_id_ascend]))
    
    
y_train_loss_df=pd.DataFrame(all_train_loss_array)
y_train_loss_df.to_csv('all_train_loss_array.csv',header=0,index=None)

y_val_loss_df=pd.DataFrame(all_val_loss_array)
y_val_loss_df.to_csv('all_val_loss_array.csv',header=0,index=None)    

# y_pred_value_df=pd.DataFrame(y_pred_value).T
# y_pred_value_df.to_csv('y_train_value.csv',header=0,index=None)

# y_test_value_df=pd.DataFrame(y_test_value).T
# y_test_value_df.to_csv('y_test_value.csv',header=0,index=None)

# y_train_value_df=pd.DataFrame(y_train_value).T
# y_train_value_df.to_csv('y_train_value.csv',header=0,index=None)

# y_train_pred_value_df=pd.DataFrame(y_train_pred_value).T
# y_train_pred_value_df.to_csv('y_train_pred_value.csv',header=0,index=None)

# y_test_r_2_value_df=pd.DataFrame(y_test_r_2_value).T
# y_test_r_2_value_df.to_csv('y_test_r_2_value.csv',header=0,index=None)

# y_test_rmse_value_df=pd.DataFrame(y_test_rmse_value).T
# y_test_rmse_value_df.to_csv('y_test_rmse_value.csv',header=0,index=None)

# y_train_r_2_value_df=pd.DataFrame(y_train_r_2_value).T
# y_train_r_2_value_df.to_csv('y_train_r_2_value.csv',header=0,index=None)

# y_train_rmse_value_df=pd.DataFrame(y_train_rmse_value).T
# y_train_rmse_value_df.to_csv('y_train_rmse_value.csv',header=0,index=None)