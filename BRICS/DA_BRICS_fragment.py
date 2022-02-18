# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:46:50 2020

@author: osberttan
"""

from rdkit import Chem
import BRICS_environment as BRICS_env
import DA_fragment as DA_frag
import re




#fragment smiles database
acceptor_smiles_DB = DA_frag.A_smiles_vals
bridge_smiles_DB = DA_frag.B_smiles_vals
donor_smiles_DB = DA_frag.D_smiles_vals

###############################
#acceptor BRICS env match and fragment creation
comp_acceptor_BRICSfrag = {}            #comprehensive frag key
acceptor_BRICSfrag = {}                 #smiles as frag key


for A_moity_smi in acceptor_smiles_DB:
    A_moity = Chem.MolFromSmiles(A_moity_smi)
    A_moity = Chem.AddHs(A_moity)
    
    #match BRICS env
    for BRICSenv_id, patt in BRICS_env.environMatchers.items():
        match_res = A_moity.HasSubstructMatch(patt)
        
        if match_res == True:
            match_index = A_moity.GetSubstructMatches(patt)
            
            #create BRICS-like fragment (1 entry)
            for entry_index in match_index:
                entry_atomic_id = entry_index[0]
                
                #find adjescent H to make dummy atom
                entry_atom = A_moity.GetAtomWithIdx(entry_atomic_id)
                
                for x in entry_atom.GetNeighbors():
                    if x.GetAtomicNum() == 1:
                        substitute_H_index = x.GetIdx()
                        
                        #set H isotope for atom to be substituted
                        substituting_atom = Chem.Atom(0)
                        
                        env_No = int(re.search(r'\d+', BRICSenv_id).group())
                        substituting_atom.SetIsotope(env_No)
                        substituting_atom.SetNoImplicit(True)
                        
                        #replacing with dummy atom
                        A_moity_W = Chem.RWMol(A_moity)
                        A_moity_W.ReplaceAtom(substitute_H_index,substituting_atom)
                        
                        #making record for BRICS frag
                        fragsmile_A_moity = Chem.MolToSmiles(A_moity_W)
                        frag_A_moity = Chem.MolFromSmiles(fragsmile_A_moity)
                        
                        comp_frag_key = (fragsmile_A_moity, BRICSenv_id, entry_atomic_id)   #(smile, BRICS env, entry index)
                        frag_key = fragsmile_A_moity                                        #smile as key
                        
                        comp_acceptor_BRICSfrag[comp_frag_key] = frag_A_moity
                        acceptor_BRICSfrag[frag_key] = frag_A_moity
                        



############################################################################
#donor BRICS env match and fragment creation
comp_donor_BRICSfrag = {}            #comprehensive frag key
donor_BRICSfrag = {}                 #smiles as frag key

    
for D_moity_smi in donor_smiles_DB:
    D_moity = Chem.MolFromSmiles(D_moity_smi)
    D_moity = Chem.AddHs(D_moity)
    
    
    #match BRICS env
    for BRICSenv_id, patt in BRICS_env.environMatchers.items():
        match_res = D_moity.HasSubstructMatch(patt)
        
        if match_res == True:
            match_index = D_moity.GetSubstructMatches(patt)
            
            #create BRICS-like fragment (1 entry)
            for entry_index in match_index:
                entry_atomic_id = entry_index[0]
                
                #find adjescent H to make dummy atom
                entry_atom = D_moity.GetAtomWithIdx(entry_atomic_id)
                
                for x in entry_atom.GetNeighbors():
                    if x.GetAtomicNum() == 1:
                        substitute_H_index = x.GetIdx()
                        
                        #set H isotope for atom to be substituted
                        substituting_atom = Chem.Atom(0)
                        
                        env_No = int(re.search(r'\d+', BRICSenv_id).group())
                        substituting_atom.SetIsotope(env_No)
                        substituting_atom.SetNoImplicit(True)
                        
                        #replacing with dummy atom
                        D_moity_W = Chem.RWMol(D_moity)
                        D_moity_W.ReplaceAtom(substitute_H_index,substituting_atom)
                        
                        #making record for BRICS frag
                        fragsmile_D_moity = Chem.MolToSmiles(D_moity_W)
                        frag_D_moity = Chem.MolFromSmiles(fragsmile_D_moity)
                        
                        comp_frag_key = (fragsmile_D_moity, BRICSenv_id, entry_atomic_id)   #(smile, BRICS env, entry index)
                        frag_key = fragsmile_D_moity                                        #smile as key
                        
                        comp_donor_BRICSfrag[comp_frag_key] = frag_D_moity
                        donor_BRICSfrag[frag_key] = frag_D_moity





############################################################################



for B_moity_smi in bridge_smiles_DB:
    B_moity = Chem.MolFromSmiles(B_moity_smi)
    B_moity = Chem.AddHs(B_moity)
    
    

