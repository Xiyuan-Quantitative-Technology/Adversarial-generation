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

# with open('ECFP_qm9_dataset.pkl', 'rb') as f:
#     data_list = pickle.load(f)
#     f.close()
    

# tasks, train_dataset, val_dataset, test_dataset, transformers = data_list
# train_smiles = train_dataset.ids
# val_smiles = val_dataset.ids
# test_smiles = test_dataset.ids
# data_list = [tasks, train_smiles, val_smiles, test_smiles]

# with open('ECFP_qm9_data_7_11.pkl', 'wb') as f:
#     pickle.dump(data_list, f)
#     f.close()
#load dataset
# with open('ECFP_QM9_dataset.pkl', 'rb') as f:
#     data_list = pickle.load(f)
#     f.close()
    

# tasks, train_dataset, val_dataset, test_dataset, transformers = data_list

#########################################################################
# train_smiles = train_dataset.ids
# val_smiles = val_dataset.ids
# test_smiles = test_dataset.ids

# with open('ECFP_qm9_data.pkl', 'rb') as f:
#     data_list = pickle.load(f)
#     f.close()
with open('d1a1_all_smile.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
    
    
all_smile = data_list    
    
# tasks, train_smiles, val_smiles, test_smiles = data_list
with open('fullD1A1_pred_ECFP_input.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
    
    
input_smile, M_ECFP = data_list

all_smile = []
for i in input_smile:
    all_smile.append(str(i[0]))
# X_mol_idseq, X_char_vocab = build_id_seq(input_smile)
with open('D1A1_database_input.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
    
    
tasks, x, test_smiles = data_list

for i in x:
    all_smile.append(str(i))
    
all_smile=np.array(all_smile)
test_X_mol_idseq, test_X_char_vocab = build_id_seq(all_smile)


# with open('d1a1_all_smile.pkl', 'wb') as f:
#     pickle.dump(all_smile, f)
#     f.close()
# index_id = list(range(0,len(xx)))
# np.random.shuffle(index_id)

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

 
# all_loss_array = np.array(all_val_loss)
# sorted_id = all_loss.index(np.min(all_loss_array))

# used_model = all_model[sorted_id]

# latent_space = used_model.encoder_forward(test_X_mol_idseq)
# a,b,c  = used_model.decoder_forward(test_X_mol_idseq, latent_space)

# c=c.cpu().detach().numpy()
# bb=[]
# m = torch.nn.Softmax(dim=0)

# for i in range(len(b)):
#     q = m(b[i,:]).cpu().detach().numpy().tolist()
#     bb.append(q.index(np.max(q)))
    
# index_22 = [i for i,x in enumerate(c.tolist()) if x==22]
# index_23 = np.array([i for i,x in enumerate(c.tolist()) if x==23])

# index_23_ = []

# for i in index_22:
#    l = index_23[index_23 < i]
#    if len(l)==0:
#        index_23_.append(0)
#    else:
#        index_23_.append(l[-1])

# true_xx = []

# pred_yy = []

# for i,j in zip(index_23_[1:],index_22[1:]):
#    true_xx.append(c[i+1:j])
#    pred_yy.append(bb[i+1:j])
   
# same_num = []

# for i in range(len(true_xx)):
#    aaa = np.sum(true_xx[i] - pred_yy[i])
#    if aaa == 0:
#        same_num.append(1)
#    else:
#        same_num.append(0)
       
       
# precision = sum(same_num) / len(true_xx)

#dict_i2c = X_char_vocab.i2c
#all_smi = []
#for i in pred_yy:
#    a=''
#    smi=[dict_i2c[k] for k in i]
#    aa=a.join(smi)
#    all_smi.append(aa)
#    
#eff_smi = []
#for i in all_smi:
#    
#    mol = Chem.MolFromSmiles(i) 
#    try:
#        Chem.SanitizeMol(mol)
#        eff_smi.append(1)
#    except Exception as e:
#        # print(e)
#        eff_smi.append(0)
# 
#eff_rate = sum(np.array(eff_smi)) / len(all_smi)
# all_model_result = all_model

# with open('all_model.pkl', 'wb') as f:
#     pickle.dump(all_model_result, f)
#     f.close()


    
# # AAE training
# N_epoch = 10

# for i in range(N_epoch):
#     time_start = time.time()
#     loss_train = trainer.train(X_mol_idseq, recon_ratio)
#     time_end = time.time()
    
#     print(i, loss_train)
#     print(time_end-time_start)
    
    
# # AAE generation
# generated_samples = model.samples_generation(1000)


