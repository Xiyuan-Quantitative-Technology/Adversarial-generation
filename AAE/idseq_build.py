# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:58:06 2021

@author: tanzheng
"""

from define_vocabulary import CharVocab, char_set
import torch


def build_id_seq(X_mol_smiles):
    No_mol = len(X_mol_smiles)
    
    #build vocabulary
    X_char_set = char_set(X_mol_smiles)
    X_char_vocab = CharVocab(X_char_set)
    
    #transform to id sequences
    X_mol_idseq = []
    for i in range(0, No_mol):
        mol_idseq = X_char_vocab.string2ids(X_mol_smiles[i],add_bos=True, add_eos=True)
        mol_idseq = torch.tensor(mol_idseq)
        X_mol_idseq.append(mol_idseq)
    
    return X_mol_idseq, X_char_vocab
    