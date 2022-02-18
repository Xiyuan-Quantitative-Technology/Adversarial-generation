# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 22:29:51 2021

@author: tanzheng
"""

class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'
    
    
class CharVocab:
    def __init__(self, chars_set, st=SpecialTokens):
        all_symbols = sorted(list(chars_set)) + [st.bos, st.eos, st.pad, st.unk]
        
        self.st = st
        
        self.c2i = {c: i for i, c in enumerate(all_symbols)}
        self.i2c = {i: c for i, c in enumerate(all_symbols)}
        
    @property
    def bos_id(self):
        return self.c2i[self.st.bos]

    @property
    def eos_id(self):
        return self.c2i[self.st.eos]

    @property
    def pad_id(self):
        return self.c2i[self.st.pad]

    @property
    def unk_id(self):
        return self.c2i[self.st.unk]
    
    def char2id(self, char):
        if char not in self.c2i:
            return self.unk_id

        return self.c2i[char]
    
    def id2char(self, id):
        if id not in self.i2c:
            return self.st.unk

        return self.i2c[id]
    
    ###################################################################
    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids
    
    
    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos_id:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos_id:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string
    
    
###################################################################
def char_set(string_set):
    charset = set()
    
    for string in string_set:
        charset.update(string)
        
    return charset



