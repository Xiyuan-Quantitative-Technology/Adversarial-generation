# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:27:05 2020

@author: osberttan
"""

import DA_BRICS_fragment as DA_BRICS_frag
import D1_A1_builder as DA_build
import re
from rdkit import Chem
from rdkit.Chem import Draw
import pickle



acceptor_frag = DA_BRICS_frag.acceptor_BRICSfrag
donor_frag = DA_BRICS_frag.donor_BRICSfrag

D1_A1_database = {}

for A_smile, A_frag in acceptor_frag.items():
    for D_smile, D_frag in donor_frag.items():
        
        new_DAMol_item = DA_build.D1_A1_complex_builder(A_frag, D_frag)
        
        if len(new_DAMol_item) != 0:
            temp_A_smile = A_smile.replace('*', '%')
            A_BRICS_env = re.findall(r"(\d+)%",temp_A_smile)
            A_BRICS_env = 'A' + A_BRICS_env[0]
            
            temp_D_smile = D_smile.replace('*', '%')
            D_BRICS_env = re.findall(r"(\d+)%",temp_D_smile)
            D_BRICS_env = 'D' + D_BRICS_env[0]
            
            BRICS_rule = A_BRICS_env + '-' + D_BRICS_env
            
            ######################################
            new_DAMol_dictitem = list(new_DAMol_item.items())
            
            new_DAMol_smiles = new_DAMol_dictitem[0][0]
            new_DAMol = new_DAMol_dictitem[0][1]
            
            new_DAMol_key = (new_DAMol_smiles, BRICS_rule)
            
            #build DA mol database with smiles and BRICS rule
            D1_A1_database[new_DAMol_key] = new_DAMol
            
        
            
##############################################################
#draw bundles of molecules
database_smiles_rule = list(D1_A1_database.keys())
database_Mol = list(D1_A1_database.values())


#show representative molecules
Mol_Series = database_Mol[:16]
Mol_labels = database_smiles_rule[:16]

#img=Draw.MolsToGridImage(Mol_Series,molsPerRow=4,subImgSize=(200,200),legends=[str(x) for x in Mol_labels])

#img.save('DA_molecules_grid1.png') 


##############################################################
#save to pickle
file = open('full_D1_A1_database.pkl', 'wb')

pickle.dump(D1_A1_database, file)

file.close()




