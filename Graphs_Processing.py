# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:15:26 2024

@author: hazem
"""

#General
from itertools import compress
import collections
import time
import os,re,random,json
from tqdm import tqdm
import itertools
import copy
import argparse


#General
from itertools import compress
import collections
import time
import os,re,random
from tqdm import tqdm
import itertools
import copy
import ast

#pandas, numpy and sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import json


#pytorch tools
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.optim as optim
from torch_geometric.utils import to_networkx
from torch_geometric.data import Dataset, Data, download_url
from torch_geometric.loader import DataLoader
from torch_geometric.data import DataLoader
from datasets import list_datasets, load_dataset, list_metrics, load_metric
import nltk

#NLTK
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
from nltk.corpus import wordnet


#Bert
from transformers import BertModel #Bert Embedding
from transformers import AutoTokenizer #BertTokenizer
from sentence_transformers import SentenceTransformer, util



import gensim
from gensim.models import Word2Vec
import warnings
import gensim.downloader as api
from Graph_Optimization.Graph_Optimization import *


#rouge score
from rouge_score import rouge_scorer

from tools.logger import *
def Running_time(t, desc, p, k):
    elapsed_time = time.time() - t
    progress = round((p / k) * 100, 2)
    print(f"\033[2K\rRunning Time: {int(elapsed_time // 3600)} hours {int((elapsed_time // 60) % 60)} mins {int(elapsed_time % 60)} secs || Number of {desc}: {p}/{k} ({progress}%)", end="", flush=True)
def Line():
    print("\033[1m====================================\033[0m")
    print('\n\n')
    
def clog(s):
    logger.info(s)
    print("________________________________________")



def main():
    parser = argparse.ArgumentParser(description='Preprocessing dataset')

    parser.add_argument('--dataset', type=str, default='Arxiv', help='The dataset directory.')
    parser.add_argument('--task', type=str, default='train', help='dataset [train|validation|test]')
    
    args = parser.parse_args()
    
    
    
    path='datasets/'+args.dataset+'/'+args.task+'/'
    #__________________________________________________
    clog("Step 1: Load Cleaned Sentences")
    
    with open(path+'Cleaned_Sentences.json', 'r') as file:
        Cleaned_Sentences_Tokenized = json.load(file)
    print('Cleaned Articles Sentences Were Loaded Successfully')
    Line()
    if args.dataset=='Arxiv':
        clog("Additional Step: Modify the Sentences length")
        #total Sentence Length
        CSL=sum([len(x) for x in Cleaned_Sentences_Tokenized])
        print('The total Number of Cleaned Corpus Sentences : ',CSL)
        
        Cleaned_Sentences_Tokenized=[x[:150] for x in Cleaned_Sentences_Tokenized]
        print('After Shrinking : ')
        CSL=sum([len(x) for x in Cleaned_Sentences_Tokenized])
        print('The total Number of Cleaned Corpus Sentences : ',CSL)
    
    #__________________________________________________
    clog("Step 2: Generate Node Sentences")
    SENTENCES_nodes=[]
    p=0
    t=time.time()
    for article in Cleaned_Sentences_Tokenized:
        nodes=create_sentence_node(article)
        SENTENCES_nodes.append(nodes)
        p+=1
        Running_time(t,'Creating Sentence nodes',p,len(Cleaned_Sentences_Tokenized)) if p%50==0 else None
    Running_time(t,'Creating Sentence nodes',p,len(Cleaned_Sentences_Tokenized))
    sentences_number=sum([len(x) for x in SENTENCES_nodes])
    print("\nThe Number of Sentences in all Articles : ",sentences_number)
    Line()
    #___________________________________
    clog("Step 3: Generate Adjecancy Sentences edges")
    SENTENCES_edges=[]
    p=0
    t=time.time()
    for r in SENTENCES_nodes:
        edge = [[r[i], r[i + 1]] for i in range(len(r) - 1)]
        SENTENCES_edges.append(edge)
        p+=1
        Running_time(t,'Creating Sentence edges',p,len(SENTENCES_nodes)) if p%50==0 else None
    Running_time(t,'Creating Sentence edges',p,len(SENTENCES_nodes))
    print("")
    Line()
    
    #___________________________________
    clog("Step 4: Check if the articles have repeated sentences")
    
        #repeated sentences means [a,b,c,d,a] a is repeated
    a=[True if len(x)!=len(set(x)) else False for x in Cleaned_Sentences_Tokenized  ]
    print("The number of articles have repeated sentences :",sum(a))
    b=[not x for x in a]
    print("The number of articles have no repeated sentences :",sum(b))
    print('The number of all articles :',len(Cleaned_Sentences_Tokenized))
    Line()
    #__________________________________
    
    clog("Step 5: Collect Equality pair in each article : e.g. [(s1,s8),(s15,s42)]")
    
    #article equality pairs
    articles_EP=[]
    t=time.time()
    p=0
    for nodes,article_sentences in zip(SENTENCES_nodes,Cleaned_Sentences_Tokenized):
        equal_pairs=collect_equal_pairs(nodes,article_sentences)
        articles_EP.append(equal_pairs)
        p+=1
        Running_time(t,'equality pair',p,len(SENTENCES_nodes)) if p%250==0 else None
    Running_time(t,'equality pair',p,len(SENTENCES_nodes))
    print("")
    Line()
    #____________________________________
    clog("Step 6: Updating Edges according to the equality pairs")
    t=time.time()
    p=0
    #contain the edges after distinguish equality sentences
    updated_edges=[]
    for EP,edges in zip(articles_EP,SENTENCES_edges):
        if len(EP)!=0:
            article_edges=updating_edges(EP,edges)
        else:
            article_edges=[(edge[0],edge[1]) for edge in edges]
        updated_edges.append(article_edges)
        p+=1
        Running_time(t,'Update edges',p,len(SENTENCES_nodes)) if p%250==0 else None
    Running_time(t,'Update edges',p,len(SENTENCES_nodes))
    print("")
    Line()
    #_____________________________________
    clog('Step 7: Deleting Extra Nodes')
    updated_nodes=[]
    print('Total Sentences Before Filtering : ',sum([len(x) for x in SENTENCES_nodes]))
    for nodes,EP in zip(SENTENCES_nodes,articles_EP):
        if len(EP)!=0:
            new_nodes=delete_extra_nodes(nodes,EP)
        else:
            new_nodes=nodes
        updated_nodes.append(new_nodes)
    del SENTENCES_nodes,a
    print('Total Sentences (updated_nodes) After Filtering : ',sum([len(x) for x in updated_nodes]))
    
    Line()
    #_____________________________________
    clog('Step 8: Filter The sentences depending on the nodes we have')
    t=time.time()
    
    p=0
    for x in range(len(Cleaned_Sentences_Tokenized)):
        Cleaned_Sentences_Tokenized[x]=list(pd.Series(Cleaned_Sentences_Tokenized[x])[updated_nodes[x]])
        p+=1
        Running_time(t,'filter sentences',p,len(Cleaned_Sentences_Tokenized)) if p%250==0 else None
    Running_time(t,'filter sentences',p,len(Cleaned_Sentences_Tokenized))
    print("")
    Line()
    
    clog('Step 9: Save the Cleaned Sentences')
    with open(path+'Cleaned_Sentences.json', 'w') as json_file:
        json.dump(Cleaned_Sentences_Tokenized, json_file)
    print('Saved Cleaned Tokenized Sentences')
    
    del Cleaned_Sentences_Tokenized
    print('Deleted Cleaned Sentences Object')
    Line()
    clog('Step 10: Load Tokenized Sentences')
    with open(path+'Sentences.json', 'r') as file:
        Sentences_tokenized = json.load(file)
    print('Articles Sentences were Loaded Successfully')
    print('______________________________________')
    print('Shrink Tokenized Sentences')
    if args.dataset=='Arxiv':
        Sentences_tokenized=[x[:150] for x in Sentences_tokenized]
    SL=sum([len(x) for x in Sentences_tokenized])
    print('The total Number of Corpus Sentences : ',SL)
    
    print('Filter Tokenized Sentences')
    p=0
    for x in range(len(Sentences_tokenized)):
        Sentences_tokenized[x]=list(pd.Series(Sentences_tokenized[x])[updated_nodes[x]])
        p+=1
        Running_time(t,'filter sentences',p,len(Sentences_tokenized)) if p%250==0 else None
    SL=sum([len(x) for x in Sentences_tokenized])
    print('The Final total Number of Corpus Sentences : ',SL)
    
    
    #___________________________________
    clog("Step 9: Standrize Edges (Remove the gap)")
    
    SENTENCES_standrized_edges=[standrize_edges(edges) if len(EP)!=0 else edges for edges,EP in zip(updated_edges,articles_EP)]
    #standerd_edges
    print("The Length of articles Edges :",len(SENTENCES_standrized_edges))
    Line()
    #___________________________________
    clog('Step 10: Standrize Nodes')
    SENTENCES_standrized_nodes=[create_sentence_node(sentences) for sentences in Sentences_tokenized]
    Line()
    #__________________________________
    clog('Step 11: Save the data')
    print('Step 11.1: Update saving the sentences')
    with open(path+'Sentences.json', 'w') as json_file:
        json.dump(Sentences_tokenized, json_file)
    print('Saved Tokenized Sentences')
    print("________________________________________")
   
    print('Step 10.2: Save Nodes and Edges')
    with open(path+'Sentences_Nodes.json', 'w') as json_file:
        json.dump(SENTENCES_standrized_nodes, json_file)
    print('Saved Sentences Nodes')
    print("________________________________________")
    with open(path+'Sent_adj/'+'Edges.json', 'w') as json_file:
        json.dump(SENTENCES_standrized_edges, json_file)
    print('Saved Sentences Edges')
    print("________________________________________")
    
    


main()