# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:06:51 2020

@author: osberttan
"""

#reading DA fragment

#from os.path import join, dirname, abspath
import xlrd
#from xlrd.sheet import ctype_text


# Open the workbook
xl_workbook = xlrd.open_workbook('D:/tanzheng/AI for OLED design/Donor acceptor search space/DA fragment.xlsx')

sheet_names = xl_workbook.sheet_names()

A_sheet = xl_workbook.sheet_by_name('Acceptor')
B_sheet = xl_workbook.sheet_by_name('Bridge')
D_sheet = xl_workbook.sheet_by_name('Donor')

#####################################################
No_headerline = 3

#read acceptor smiles
A_smiles_vals = []

for row_id in range(No_headerline, A_sheet.nrows):
    cell_obj = A_sheet.cell(row_id, 0)
    A_smiles_vals.append(cell_obj.value)
    
    
#read Bridge smiles
B_smiles_vals = []

for row_id in range(No_headerline, B_sheet.nrows):
    cell_obj = B_sheet.cell(row_id, 0)
    B_smiles_vals.append(cell_obj.value)
    
    
#read Donor smiles
D_smiles_vals = []

for row_id in range(No_headerline, D_sheet.nrows):
    cell_obj = D_sheet.cell(row_id, 0)
    D_smiles_vals.append(cell_obj.value)
    
    
    
    
