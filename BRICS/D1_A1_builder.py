# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:20:57 2020

@author: osberttan
"""

from rdkit import Chem
from rdkit.Chem import BRICS

'''
acceptor_frag = DA_BRICS_frag.acceptor_BRICSfrag['[10*]N1C(=O)C([H])=C([H])C1=O']
donor_frag = DA_BRICS_frag.donor_BRICSfrag['[16*]c1c([H])c([H])c([H])c([H])c1N(C([H])([H])[H])C([H])([H])[H]']
'''


def D1_A1_complex_builder(acceptor_frag, donor_frag):
    
    ###############################
    # 1D-1A combination
    #sythesize molecule
    
    self_A_sythetic_M = BRICS.BRICSBuild([acceptor_frag], maxDepth=0)
    self_D_sythetic_M = BRICS.BRICSBuild([donor_frag], maxDepth=0)
    
    sythetic_M = BRICS.BRICSBuild([acceptor_frag,donor_frag], maxDepth=0)
    
    selfA_sythetic_prods = []
    selfD_sythetic_prods = []
    sythetic_prods = []
    
    max_No_prods = 10
    
    for x in range(max_No_prods):
        try:
            selfA_sythetic_prods.append(next(self_A_sythetic_M))
        except:
            break
    
    for x in range(max_No_prods):
        try:
            selfD_sythetic_prods.append(next(self_D_sythetic_M))
        except:
            break
    
    for x in range(max_No_prods):
        try:
            sythetic_prods.append(next(sythetic_M))
        except:
            break
    
    
    ########################################################
    #doing sanitization
    '''
    for mol in selfA_sythetic_prods + selfD_sythetic_prods + sythetic_prods:
        Chem.SanitizeMol(mol)
    '''  
    
    
    #build smiles dictionary
    selfA_sythetic_dict = {}
    selfD_sythetic_dict = {}
    sythetic_dict = {}
    
    for mol in selfA_sythetic_prods:
        selfA_sythetic_dict[Chem.MolToSmiles(mol)] = mol
    
    for mol in selfD_sythetic_prods:
        selfD_sythetic_dict[Chem.MolToSmiles(mol)] = mol
        
    for mol in sythetic_prods:
        sythetic_dict[Chem.MolToSmiles(mol)] = mol
    
    
    
    #remove self-sythetic molecules
    for smi_A in list(selfA_sythetic_dict.keys()):
        if smi_A in list(sythetic_dict.keys()):
            del sythetic_dict[smi_A]
    
    for smi_D in list(selfD_sythetic_dict.keys()):
        if smi_D in list(sythetic_dict.keys()):
            del sythetic_dict[smi_D]
    


    ########################################################
    #return sythetic molecule 
    if len(sythetic_dict) == 1:
        print("success sythesis")
        return sythetic_dict
    
    elif len(sythetic_dict) == 0:
        print('sythesis not accessible')
        return {}
    
    else:
        print('retrospect sythesis')
        return {}        
        
    
    
