# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 22:55:01 2021

@author: tanzheng
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab, embedding_layer, encod_hidden_D, lat_v_D, bidir_flag):
         super(Encoder, self).__init__()
         
         self.vocab = vocab
         self.embedding_layer = embedding_layer
         
         # Encoder
         self.GRU_encoder = nn.GRU(embedding_layer.embedding_dim,
                                  encod_hidden_D,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=bidir_flag)
         
         # dimension of the last hidden state in GRU sequence
         last_hidden_D = encod_hidden_D * (2 if bidir_flag else 1)
         
         # encoder latent linear layer
         self.encod_hid2lat_layer = nn.Linear(last_hidden_D, lat_v_D)
         
    def forward(self, x):
        """Encoder step
        :param x: list of tensors of longs, input sequences x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        """
        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.vocab.pad_id)
        
        embeded_x = self.embedding_layer(x)
        
        _, h_n = self.GRU_encoder(embeded_x)
        
        # last hidden state for sequences
        h_n = h_n[-(1 + int(self.GRU_encoder.bidirectional)):]
        h_n = torch.cat(h_n.split(1), dim=-1).squeeze(0)
        
        #latent space vectors
        z = self.encod_hid2lat_layer(h_n)
        
        return z
    

class Decoder(nn.Module):
    def __init__(self, vocab, embedding_layer, decod_hidden_D, lat_v_D):
        super(Decoder, self).__init__()
        
        self.vocab = vocab
        n_vocab_item = len(self.vocab.c2i)
        self.embedding_layer = embedding_layer
        
        # decoder latent to hidden layer
        self.decod_lat2hid_layer = nn.Linear(lat_v_D, decod_hidden_D)
        
        # Decoder
        self.GRU_decoder =  nn.GRU(embedding_layer.embedding_dim + lat_v_D,
                                   decod_hidden_D,
                                   num_layers=3,
                                   batch_first=True,
                                   bidirectional=False)
        
        # decoder decipher layer
        self.decod_deciph_layer = nn.Linear(decod_hidden_D, n_vocab_item)
        
    def forward(self, x, z):
        """Decoder step, emulating x ~ G(z)
        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon loss
        """
        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.vocab.pad_id)
        
        embeded_x = self.embedding_layer(x)
        
        z_0 = z.unsqueeze(1).repeat(1, embeded_x.size(1), 1)
        x_input = torch.cat([embeded_x, z_0], dim=-1)
        
        h_0 = self.decod_lat2hid_layer(z)
        h_0 = h_0.unsqueeze(0).repeat(self.GRU_decoder.num_layers, 1, 1)
        
        output, _ = self.GRU_decoder(x_input, h_0)
        y = self.decod_deciph_layer(output)
        
        true_x = x[:, 1:].contiguous().view(-1)
        true_y = y[:,:-1,:].contiguous().view(-1, y.size(2))
                                  
        # reconstruction loss
        recon_loss = F.cross_entropy(true_y, true_x, ignore_index=self.vocab.pad_id)
        
        return recon_loss


class Discriminator(nn.Module):
    def __init__(self, lat_v_D, dis_hid_D):
        super(Discriminator, self).__init__()
        
        # Discriminator classifier
        self.discriminator_layer = nn.Sequential(nn.Linear(lat_v_D, dis_hid_D),
                                                 nn.ReLU(),
                                                 nn.Linear(dis_hid_D, int(dis_hid_D/2)),
                                                 nn.ReLU(),
                                                 nn.Linear(int(dis_hid_D/2), 1),
                                                 nn.Sigmoid())
        
    def forward(self, input):
        class_prob = self.discriminator_layer(input)
        return class_prob
    
    
#####################################################################
class AAE(nn.Module):
    def __init__(self, vocab, emb_D, encod_hidden_D, decod_hidden_D, lat_v_D, bidir_flag, dis_hid_D):
        super(AAE, self).__init__()
        
        self.vocab = vocab
        self.latent_size = lat_v_D
        
        # character embeddings layer
        n_vocab_item = len(self.vocab.c2i)
        self.embedding_layer = nn.Embedding(n_vocab_item, emb_D, 
                                            padding_idx=self.vocab.pad_id)
        
        # Encoder
        self.encoder = Encoder(self.vocab, self.embedding_layer, \
                               encod_hidden_D, lat_v_D, bidir_flag)
        
        # Decoder
        self.decoder = Decoder(self.vocab, self.embedding_layer, \
                               decod_hidden_D, lat_v_D)
        
        # Discriminator
        self.discriminator = Discriminator(lat_v_D, dis_hid_D)
                                                
        
    ##################################################################################
    def encoder_forward(self, x):
        return self.encoder.forward(x)

    def decoder_forward(self, x, z):
        return self.decoder.forward(x, z)

    def discriminator_forward(self, z):
        return self.discriminator.forward(z)
    
    ##################################################################################
    def random_sample_latent(self, n):
        return torch.randn(n, self.latent_size)
    
    def samples_generation(self, No_sam, max_len=100):
        with torch.no_grad():
            ran_z = self.random_sample_latent(No_sam)
            ran_z_0 = ran_z.unsqueeze(1)
            
            # Initial values
            h = self.decoder.decod_lat2hid_layer(ran_z)
            h = h.unsqueeze(0).repeat(self.decoder.GRU_decoder.num_layers, 1, 1)
            
            write_char = torch.tensor(self.vocab.bos_id).repeat(No_sam)
            x = torch.tensor([self.vocab.pad_id]).repeat(No_sam, max_len)
            
            x[:, 0] = self.vocab.bos_id
            end_pads_loc = torch.tensor([max_len]).repeat(No_sam)
            eos_mask = torch.zeros(No_sam, dtype=torch.bool)
            
            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.embedding_layer(write_char).unsqueeze(1)
                x_input = torch.cat([x_emb, ran_z_0], dim=-1)
                
                o, h = self.decoder.GRU_decoder(x_input, h)
                y = self.decoder.decod_deciph_layer(o.squeeze(1))
                y = F.softmax(y, dim=-1)
                
                # updating output x
                write_char = torch.multinomial(y, 1)[:, 0]
                x[eos_mask.logical_not(), i] = write_char[eos_mask.logical_not()]
                
                i_eos_mask = eos_mask.logical_not() & (write_char == self.vocab.eos_id)
                end_pads_loc[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask
                
            # Converting `x` to list of string
            new_x = []
            for i in range(x.size(0)):
                id_seq = x[i, :end_pads_loc[i]].tolist()
                string = self.vocab.ids2string(id_seq, rem_bos=True, rem_eos=True)
                
                new_x.append(string)
                
            return new_x
            
            