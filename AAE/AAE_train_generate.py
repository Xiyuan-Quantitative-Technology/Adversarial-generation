# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:42:50 2021

@author: tanzheng


"""

import torch
import pickle
import numpy as np
import time
from idseq_build import build_id_seq
from AAE_module import AAE
from AAE_trainer import AAE_Trainer


with open('d1a1_all_smile.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()  
input_smile, M_ECFP = data_list

all_smile = []
for i in input_smile:
    all_smile.append(str(i[0]))
    
all_smile=np.array(all_smile)
test_X_mol_idseq, test_X_char_vocab = build_id_seq(all_smile)

#########################################################################


#set seed and device
seed=12
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Python random module.
torch.manual_seed(seed) 

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(32)

#########################################################################
# AAE parameters
emb_D = 64
encod_hidden_D = 256
decod_hidden_D = 512
lat_v_D = 128
bidir_flag = True
dis_hid_D = 1024

#training parameters
lr, batch_size = 0.0005, 64
# weight of recon loss
recon_ratio = 0.5

#model generation
model = AAE(test_X_char_vocab, emb_D, encod_hidden_D, decod_hidden_D, \
            lat_v_D, bidir_flag, dis_hid_D)
trainer = AAE_Trainer(model, lr, batch_size)

#
N_epoch = 60
all_loss = []

all_model = []
norm_ = np.ceil(len(test_X_mol_idseq) / batch_size)
# all_val_loss = []
for i in range(N_epoch):
    strat_time = time.time()
    loss_train, each_model = trainer.train(test_X_mol_idseq, recon_ratio)
    end_time = time.time()
    print(end_time - strat_time)
    all_loss.append(loss_train)
    all_model.append(each_model)
    print(i, loss_train['D_loss']/norm_, loss_train['G_loss']/norm_, loss_train['recon_loss']/norm_)
    
all_model_result = [all_model,all_loss]

with open('all_mode_loss_0.0005 dis1024 0.5 batch64 60 full.pkl', 'wb') as f:
    pickle.dump(all_model_result, f)
    f.close()
# AAE generation
generated_samples = model.samples_generation(1000)   

 



