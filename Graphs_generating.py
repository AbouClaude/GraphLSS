# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 01:08:14 2024

@author: hazem
"""

import time
import argparse
from Graph_Embeddings import *
from tools.logger import *

def Line():
    print("\033[1m====================================\033[0m")
    print('\n\n')
def clog(s):
    logger.info(s)
    print("________________________________________")
    


        

def main():
    tg=time.time()
    parser = argparse.ArgumentParser(description='Preprocessing dataset')
    
    parser.add_argument('--dataset', type=str, default='Arxiv', help='The dataset directory.')
    parser.add_argument('--task', type=str, default='train', help='dataset [train|validation|test]')
    
    args = parser.parse_args()
    
    
    
    path='datasets/'+args.dataset+'/'+args.task+'/'
   
    
    print('The data has been readed successfully.')
    print('_________________________________________')
    clog('Step 2: Create the graphs objects that will be used in the model.')
    graph_path='Graphs/'+args.dataset+'/'+args.task+'/'
    print('Graph Path : ', graph_path)
    
    Creating_Graphs_objects(path,graph_path)
    print("")
    print("All Graphs have been created and saved successfully")
    print("_________________________________________")
    print('\n\n')
    a=elapsed_time(tg)
    print('Creating Graphs takes : ',tg)
    
    
main()
    