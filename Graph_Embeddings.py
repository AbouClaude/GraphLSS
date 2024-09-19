# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:24:17 2024

@author: hazem
"""

import time
import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer, util
import json
import gensim.downloader as api


def Running_time(t, desc, p, k):
    elapsed_time = time.time() - t
    progress = round((p / k) * 100, 2)
    print(f"\033[2K\rRunning Time: {int(elapsed_time // 3600)} hours {int((elapsed_time // 60) % 60)} mins {int(elapsed_time % 60)} secs || Number of {desc}: {p}/{k} ({progress}%)", end="", flush=True)
    
    
def elapsed_time(start_time):
    """
    Calculate the elapsed time from the given start time in hours, minutes, and seconds.
    
    :param start_time: The start time in seconds (from `time.time()`)
    :return: A string with hours, minutes, and seconds
    """
    # Calculate the elapsed time in seconds
    elapsed_seconds = time.time() - start_time
    
    # Convert the elapsed time to hours, minutes, and seconds
    hours = int(elapsed_seconds // 3600)  # 1 hour = 3600 seconds
    remaining_seconds = elapsed_seconds % 3600  # Seconds left after extracting hours
    
    minutes = int(remaining_seconds // 60)  # 1 minute = 60 seconds
    seconds = int(remaining_seconds % 60)  # Remaining seconds
    
    # Build the output string
    elapsed_time_str = f"{hours} hours, {minutes} minutes, and {seconds} seconds"
    
    return elapsed_time_str
    

    
def Sentences_encoding(listSentences,model=SentenceTransformer('all-MiniLM-L6-v2',device='cuda')):
    e=model.encode(listSentences, show_progress_bar=False,
                              convert_to_tensor=True)
    e=e.to('cpu')
    #e = [model.encode(s) for s in listSentences]
    return e
def Sentences_mapping(listSentences):
    return {sentence: i for i, sentence in enumerate(listSentences)}


def get_Edges(listedges):
    #make source list
    s=[x[0] for x in listedges]
    #make distination list
    d=[x[1] for x in listedges]
    #make the edges undirected
    sn=s+d
    dn=d+s
    edges=torch.tensor([sn,dn],dtype=torch.long)
    return edges

def get_hetero_Edges(listedges):
    #make source list
    s=[x[0] for x in listedges]
    #make distination list
    d=[x[1] for x in listedges]
    edges=torch.tensor([s,d],dtype=torch.long)
    return edges

def get_Edges_feature(listfeature):
    return torch.tensor([x for x in listfeature],dtype=torch.float32)

def Word2vec_Embedding(list_of_word,word2vec,uniform=0.25,k=300):
    words_embeddings=[]
    for x in list_of_word:
        try:
            words_embeddings.append(word2vec[word])
        except:
            words_embeddings.append(np.random.uniform(-1 * uniform, uniform, k).round(6).tolist())
            
    return torch.tensor(words_embeddings,dtype=torch.float32)


#LS: list of Sentences       #L: labeled Sentences           #SE: Sentence Edges
#SF: Sentence Features       #SSE: S Similarity Edges        #SSF: S Similarity Features
#W: Words list               #WS: W to S Edges               #WSF: W to S Features
#WW: W to W Edges            #WWF: W to W Features           #wv: word embedding model    index: index of the sentence
def Create_Hetereograph(LS,L,SE,SF,SSE,SSF,W,WS,WSF,WW,WWF,wv,index):
    #Sentence Node Feature
    data = HeteroData()
    data['Sentences'].x=Sentences_encoding(LS)
    data['Sentences'].y=L
    data['Sentences'].nodeidx=torch.tensor([index]*len(L))
    data['Sentences'].idx=torch.tensor(index)
    #________________________________________
    
    
    #Sentence Edges
    data['Sentences','Adjecancy','Sentences'].edge_index=get_Edges(SE)
    
    #Sentence Edges Features
    data['Sentences','Adjecancy','Sentences'].edge_attr=get_Edges_feature(SF*2)
    
    #________________________________________
    #Sentence Similarity Edges
    data['Sentences','Similarity','Sentences'].edge_index=get_Edges(SSE)
    #Sentence Similarity Features
    data['Sentences','Similarity','Sentences'].edge_attr=get_Edges_feature(SSF*2)
    
    #________________________________________
    data['Words'].x=Word2vec_Embedding(W,wv)
    #_______________________________________
    
    #________________________________________
    data['Words','similar','Words'].edge_index=get_Edges(WW)
    data['Words','similar','Words'].edge_attr=get_Edges_feature(WWF*2)
    
    
    #edges Words sentences
    
    data['Words','belong','Sentences'].edge_index=get_hetero_Edges(WS)
    
    #edges features Words Sentences
    data['Words','belong','Sentences'].edge_attr=get_Edges_feature(WSF)
    WS = [(b, a) for a, b in WS]
    #WSF=list(reversed(WSF))
    data['Sentences','Contain','Words'].edge_index=get_hetero_Edges(WS)
    data['Sentences','Contain','Words'].edge_attr=get_Edges_feature(WSF)
    
    return data


def Read_data(path,si,ei):
    
    df_graph=pd.DataFrame()
    ei+=1
    #reading Sentences and Sentences Nodes
    with open(path+'Sentences.json', 'r') as file:
        df_graph['Sentences'] = json.load(file)[si:ei]
       
    with open(path+'Sentences_Nodes.json', 'r') as file:
        df_graph['Sentences_nodes'] = json.load(file)[si:ei]

        
    #read Adjecancy edges
    with open(path+'sent_adj/'+'Edges.json', 'r') as file:
        df_graph['Sentences_edges'] = json.load(file)[si:ei]
    with open(path+'sent_adj/'+'Features.json', 'r') as file:
        df_graph['Sentences_edges_features'] = json.load(file)[si:ei]
      
        
    #read Sentence Similarity Edges
    with open(path+'Sen_sim/'+'Edges.json', 'r') as file:
        df_graph['Similarity_Edges'] = json.load(file)[si:ei]
    with open(path+'Sen_sim/'+'Features.json', 'r') as file:
        df_graph['Similarity_Features'] = json.load(file)[si:ei]
        
    #reading Sentences and Sentences Nodes
    with open(path+'Words_tokens.json', 'r') as file:
        df_graph['Words'] = json.load(file)[si:ei]
    with open(path+'Words_nodes.json', 'r') as file:
        df_graph['Words_nodes'] = json.load(file)[si:ei]
       
        
        
    #read WS Edges
    with open(path+'WS_edges/'+'Edges.json', 'r') as file:
        df_graph['Words_Sentence_Edges'] = json.load(file)[si:ei]
    with open(path+'WS_edges/'+'Features.json', 'r') as file:
        df_graph['Words_Sentence_Features'] = json.load(file)[si:ei]
     
        
    #read WW Edges
    with open(path+'WW_edges/'+'Edges.json', 'r') as file:
        df_graph['Word_Words_Edges'] = json.load(file)[si:ei]
    with open(path+'WW_edges/'+'Features.json', 'r') as file:
        df_graph['Word_Words_Features'] = json.load(file)[si:ei]
        
    with open(path+'Labels.json', 'r') as file:
        df_graph['Labels'] = json.load(file)[si:ei]
       
    df_graph.index=list(range(si,ei))
    return df_graph

'''
def read_data_with_limit(path, limit=50000):
    # Reading Summary in chunks to get the first 50,000 rows
    csv_file_path = os.path.join(path, 'dataset.csv')
    # Only load the first 50,000 rows
    df_graph = pd.read_csv(csv_file_path, usecols=['summary'], nrows=limit)

    # Load JSON files, but only get the first 50,000 elements
    json_files = {
        'Sentences': 'Sentences.json',
        'Sentences_nodes': 'Sentences_Nodes.json',
        'Sentences_edges': 'sent_adj/Edges.json',
        'Sentences_edges_features': 'sent_adj/Features.json',
        'Similarity_Edges': 'Sen_sim/Edges.json',
        'Similarity_Features': 'Sen_sim/Features.json',
        'Words': 'Words_tokens.json',
        'Words_nodes': 'Words_nodes.json',
        'Words_Sentence_Edges': 'WS_edges/Edges.json',
        'Words_Sentence_Features': 'WS_edges/Features.json',
        'Word_Words_Edges': 'WW_edges/Edges.json',
        'Word_Words_Features': 'WW_edges/Features.json',
        'Labels': 'Labels.json'
    }

    # Load the first 50,000 elements from each JSON file
    for key, file_name in json_files.items():
        json_file_path = os.path.join(path, file_name)
        with open(json_file_path, 'r') as file:
            data = json.load(file)  # Load the entire JSON
            df_graph[key] = data[:limit]
    return df_graph

def read_data_from_index(path, start_index=100001):
    # CSV: Read data from a specific index to the end
    csv_file_path = os.path.join(path, 'dataset.csv')
   
    # Load all rows from the start_index to the end
    df_graph = pd.read_csv(csv_file_path, usecols=['summary'], skiprows=range(1, start_index))  # Skip rows up to start_index
    print('data summary readed',' with length ', len(df_graph))
    # JSON: Load data from a specific index to the end
    json_files = {
        'Sentences': 'Sentences.json',
        'Sentences_nodes': 'Sentences_Nodes.json',
        'Sentences_edges': 'sent_adj/Edges.json',
        'Sentences_edges_features': 'sent_adj/Features.json',
        'Similarity_Edges': 'Sen_sim/Edges.json',
        'Similarity_Features': 'Sen_sim/Features.json',
        'Words': 'Words_tokens.json',
        'Words_nodes': 'Words_nodes.json',
        'Words_Sentence_Edges': 'WS_edges/Edges.json',
        'Words_Sentence_Features': 'WS_edges/Features.json',
        'Word_Words_Edges': 'WW_edges/Edges.json',
        'Word_Words_Features': 'WW_edges/Features.json',
        'Labels': 'Labels.json'
    }

    # Load data from each JSON file from start_index to the end
    for key, file_name in json_files.items():
        json_file_path = os.path.join(path, file_name)
        with open(json_file_path, 'r') as file:
            data = json.load(file)  # Load entire JSON content
            df_graph[key] = data[start_index-1:]  # Get data from start_index to end
            print( key, ' are readed sucessfully', 'with length : ', len(data[start_index-1:]))

    return df_graph
'''

def Creating_Graphs_objects(path,graph_path,wv = api.load('word2vec-google-news-300')):
    
    c=0
    p=0
    tg=time.time()
    for df_chunk in pd.read_csv(path+'dataset.csv',usecols=['summary'],chunksize=40000):
        print('Load Chunk : ',c)
        print('Loading graph elements')
        minidx,maxidx=min(df_chunk.index),max(df_chunk.index)
        print('Minimum Index of the chunk : ', minidx,end='      ')
        print('Maximum Index of the chunk : ',maxidx)
        td=time.time()
        df_graph=Read_data(path,minidx,maxidx)
        read_time=elapsed_time(td)
        print(f'Loading data of chunk {c}:'+read_time)
        
        print("\n\n\n")
        t=time.time()
        for idx in df_graph.index:
            
            with torch.no_grad():
                label=torch.tensor(df_graph.Labels[idx],dtype=torch.long,device='cpu')
                #print(label)
                graph=Create_Hetereograph(df_graph.Sentences[idx],label
                                    ,df_graph.Sentences_edges[idx],df_graph.Sentences_edges_features[idx]
                                    ,df_graph.Similarity_Edges[idx],df_graph.Similarity_Features[idx]
                                    ,df_graph.Words[idx]
                                    ,df_graph.Words_Sentence_Edges[idx],df_graph.Words_Sentence_Features[idx]
                                    ,df_graph.Word_Words_Edges[idx],df_graph.Word_Words_Features[idx]
                                    ,wv,idx)
            
            torch.save(graph, os.path.join(graph_path,
                                           f'graph_{idx}.pt'))
            p+=1
            
            Running_time(tg,f'Creating Graphs of chunk_{c}',p,len(df_chunk)) if p%10==0 else None
    Running_time(tg,'Creating Graphs',p,len(df_graph))
    
            
        