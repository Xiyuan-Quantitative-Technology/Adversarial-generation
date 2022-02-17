# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:53:13 2021

@author: tanzheng
"""

import numpy as np
from itertools import chain
from define_vocabulary import CharVocab, char_set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AAE_Trainer():
    def __init__(self, model, lr, batch_size):
        self.model = model
        self.lr = lr
        self.bs = batch_size
        
        self.G_optimizer = optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), \
                                      lr=self.lr, betas=(0.9,0.99))
        self.D_optimizer = optim.Adam(model.discriminator.parameters(), lr=self.lr, betas=(0.9,0.99))
        
        self.device = torch.device("cpu")
        
    def train(self, X_mol_idseq, recon_ratio):
        No_mol = len(X_mol_idseq)
        
        #epoch training
        loss_total = {'G_loss':0, 'D_loss':0, 'recon_loss':0, 'discr_real_loss':0, 
                      'real_loss':0, 'fake_loss':0}
        
        for i in range(0, No_mol, self.bs):
            #batch data prepare
            BX_mol_idseq = X_mol_idseq[i:i+self.bs]
            #BX_mol_idseq.to(self.device)
            
            temp_bs = len(BX_mol_idseq)         # temp batch size
            
            #ground truth
            real = torch.ones(temp_bs, 1)
            fake = torch.zeros(temp_bs, 1)
            
            # -----------------
            #  Train Generator
            # -----------------
            latent_space = self.model.encoder_forward(BX_mol_idseq)
            recon_loss = self.model.decoder_forward(BX_mol_idseq, latent_space)
            
            discr_output = self.model.discriminator_forward(latent_space)
            discr_real_loss = F.binary_cross_entropy(discr_output, real)
            
            G_loss = recon_ratio*recon_loss + (1-recon_ratio)*discr_real_loss
            
            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Sample noise as discriminator ground truth
            latent_ran_sample = self.model.random_sample_latent(temp_bs)
            
            discr_real_output = self.model.discriminator_forward(latent_ran_sample)
            real_loss = F.binary_cross_entropy(discr_real_output, real)
            discr_fake_output = self.model.discriminator_forward(latent_space.detach())
            fake_loss = F.binary_cross_entropy(discr_fake_output, fake)
            
            D_loss = 0.5 * (real_loss + fake_loss)
            
            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()
            
            
            loss_total['G_loss'] += G_loss.item()
            loss_total['D_loss'] += D_loss.item()
            loss_total['recon_loss'] += recon_loss.item()
            loss_total['discr_real_loss'] += discr_real_loss.item()
            loss_total['real_loss'] += real_loss.item()
            loss_total['fake_loss'] += fake_loss.item()
        return loss_total, self.model
            
        
        