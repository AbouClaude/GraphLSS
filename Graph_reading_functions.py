# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:04:07 2024

@author: hazem
"""
import pandas as pd
import os
import torch
import json
import time
from torch_geometric.loader import DataLoader
from torch_geometric.loader import HGTLoader

def Running_time(t, desc, p, k):
    elapsed_time = time.time() - t
    progress = round((p / k) * 100, 2)
    print(f"\033[2K\rRunning Time: {int(elapsed_time // 3600)} hours {int((elapsed_time // 60) % 60)} mins {int(elapsed_time % 60)} secs || Number of {desc}: {p}/{k} ({progress}%)", end="", flush=True)

def read_dataframe(path):
    df=pd.read_csv(path+"dataset.csv",usecols=['summary'])
    with open(path+'Sentences.json', 'r') as file:
        df['Sentences'] = json.load(file)
    return df

def get_graph_path(path):
    p={}
    p['train']=path+'/train/'
    p['test']=path+'/test/'
    p['validation']=path+'/validation/'
    return p

    
def Graph_train_read(files_names,path,bs=64):
    Graph_list=[torch.load(path+x) for x in files_names]
    train_loader=DataLoader(Graph_list,batch_size=bs,shuffle=True)
    #train_loader=HGTLoader(Graph_list,batch_size=64,shuffle=True)
    return train_loader

def graph_files(path):
    files= os.listdir(path)
    # Create a list of (file, modification_time) tuples
    file_timestamps = [(file, os.path.getmtime(os.path.join(path,file))) for file in files]
    sorted_files = sorted(file_timestamps, key=lambda x: x[1], reverse=False)
    sorted_files = [x[0] for x in sorted_files]
    return sorted_files

def Graph_read(files_names,path):
    Graph_list=[]
    p=0
    t=time.time()
    k=len(files_names)
    for x in files_names:
        graph=torch.load(path+x)
        Graph_list.append(graph)
        Running_time(t,'Loading Graph',p,k) if p%50==0 else None
        p+=1
    Running_time(t,'Loading Graph',p,k)
    return Graph_list