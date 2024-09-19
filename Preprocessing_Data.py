# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:53:41 2024

@author: hazem
"""

#General
import time
import warnings
import json
import argparse


#pandas, numpy and sklearn
import numpy as np
import pandas as pd

#Visualizing tools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from upsetplot import plot,UpSet
import plotly.io as pio
import plotly.graph_objects as go

from datasets import list_datasets, load_dataset, list_metrics, load_metric
import gensim
from gensim.models import Word2Vec
import warnings
import gensim.downloader as api
import seaborn as sns
import matplotlib.pyplot as plt

from tools.logger import *

from Preprocessing.Sentence_processing import *

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
    
    
    
    dataset=None
    #___________________________________________
    clog("Step 1: Import Dataset")
    if args.dataset=='Arxiv':
        dataset = load_dataset("ccdv/arxiv-summarization",trust_remote_code=True)
    elif args.dataset=='CNNDM':
        dataset= load_dataset("cnn_dailymail",'3.0.0',trust_remote_code=True)
    elif args.dataset=='Pubmed':
        dataset= load_dataset("ccdv/pubmed-summarization",trust_remote_code=True)
    print(dataset)
    for x in dataset.items():
        print(x)
        print('__________________________________')
    
    print(dataset)
    print("dataset "+ args.dataset+ " is loaded")
    Line()
   
    #___________________________________________
    clog("Step 2: Extract Information")
    t=time.time()
    articles,summaries=Extracting_information(dataset,args.task)
    Running_time(t,'Extractomg data',len(articles),len(articles))
    print("")
    Line()
    #__________________________________________
    clog("Step 3: Get the dataframe")
    
    df_train=get_dataframe(articles,summaries)
    del dataset,articles,summaries
    Line()
    #_________________________________________
    clog("Step 4: Remove Articles Duplication")
    print('Shape of the Dataframe : ',df_train.shape)
    df_train = df_train[~df_train.duplicated(subset=['article'], keep='first')]
    df_train.reset_index(drop=True,inplace=True)
    print('Shape After Remove Duplication : ',df_train.shape)
    Line()
    #__________________________________________
    clog("Step 5: Inspecting the data")
    print('Step 5.1: Articles Summaries Len relation')
    print('__________________________________')
    g=[]
    short_articles=[x for x in df_train['article'] if len(x)==0]

    print('Number of empty articles: ',len(short_articles))

    #al:articles Length
    al=pd.Series([len(x) for x in df_train.article])
    #sl:summaries length
    sl=pd.Series([len(x) for x in df_train.summary])
    #checking
    c=[True if (x-y>0) else False for x,y in zip(al,sl)]
    print('The Average of Article Length : '
          ,round(al.mean(),2),"±",round(al.std(),2))
    print('The Average of Summary Length : '
          ,round(sl.mean(),2),"±",round(sl.std(),2))
    
    print("The number of articles where they are shorter than the summaries : ", (len(c)-sum(c)))
    print("The number of articles are longer than the summaries : ",(sum(c)))
    del short_articles,c
    
    print('________________________________________________')
    print('step 5.2: Visualize Article Lengths')    
    #ags: article greater than summary
    ags=set(al.index[al>sl])
    #sga: summary greater than article
    sga=set(al.index[sl>=al])
    #a1: articles greater than 1000
    a1=set(al.index[al>1000])
    #a2: articles less than 1000
    a2=set(al.index[al<=1000])
    

    # Code where you want to suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        pcolor=["#32CD32","#FF0000"]
        dfv=form_set_df(ags,a1,a2,sga,['article > summary','article > 1000 symbols','article <= 1000 symbols','article < summary'])
        upset = UpSet(dfv,show_counts=True,facecolor='#587992',element_size=60)
    
        upset.style_subsets("article > summary", facecolor=pcolor[0])
        upset.style_subsets("article > 1000 symbols", facecolor=pcolor[0])
        upset.style_subsets("article <= 1000 symbols", facecolor=pcolor[0])
        upset.style_subsets("article < summary", facecolor=pcolor[1])
            
        upset.plot();
    
        fig = plt.suptitle("Data Inspection \n Intersection between the lengths of articles and summaries",
                 color='#00234C',size=12,weight='bold');
        plt.savefig('Figures/data_inspection_plot.png');
        del dfv,al,sl
        Line()
    
    #______________________________________________
    clog("Step 6: Filtering the data")
    desired_index=list(ags)
    fdf_train=df_train.loc[desired_index]
    fdf_train.reset_index(inplace=True,drop=True)
    fdf_train.shape

    fdf_train.head()
    print("Number of the desired articles : ",fdf_train.shape[0])
    print(fdf_train.tail(1))
    del df_train
    Line()
    
    clog("Step 7: Tokenizing Sentences")
    t=time.time()
    st=NLTK_Sentence_Tokenizer(fdf_train,'article')
    print('\n\n')
    Line()
    
    clog('Step 8: Remove "/n" and many spaces')
    
    t=time.time()
    Sentences_Tokenized=[]
    p=0
    for article in st:
        at=[]
        
        for s in article:
            s=re.sub('\n','',s)
            s=re.sub(' +',' ',s)
            at.append(s)
        p+=1
        Running_time(t, 'replacing \\n with blank', p,len(st)) if p%2000==0 else None
        Sentences_Tokenized.append(at)
    Running_time(t, 'replacing \\n with blank', p,len(st))
    del st
    print("")
    Line()
    clog('Step 9: Remove special characters ellipse')
    t=time.time()
    p=0
    for index in range(len(Sentences_Tokenized)):
        Sentences_Tokenized[index]=merge_ellipses(Sentences_Tokenized[index])
        p+=1
        Running_time(t, 'Merging ellipses " ', p,len(Sentences_Tokenized)) if p%1000==0 else None
    Running_time(t, 'Merging ellipses " ', p,len(Sentences_Tokenized))
    print("")
    Line()
    clog('Step 10: Clean Sentences (keep them both)')
    
    CST=cleaned_tokenized_sentence(Sentences_Tokenized)
    print("")
    Line()
    clog("Step 11: Check Short Sentences")
    count=0
    for x,i in zip(CST,range(0,len(CST))):
        v=[a for a in x if len(a)<10 or " " not in a]
        if len(v)!=0:
            count+=1
    print("Number of articles contain short sentences : ",count)
    check_short_sentences(CST,Sentences_Tokenized)
    del count
    Line()
    clog('Step 12: Merge the short sentences')
    t=time.time()
    p=0
    for i in range(len(Sentences_Tokenized)):
        Sentences_Tokenized[i],CST[i]=merge_prev(Sentences_Tokenized[i],CST[i])
        p+=1
        #print(p)
        Running_time(t, 'merge sentences', p,len(Sentences_Tokenized)) if p%200==0 else None
    Running_time(t, 'merge sentences', p,len(Sentences_Tokenized))
    print("")
    print('________________________________')
    print('Check the short sentences after merging')
    check_short_sentences(CST, Sentences_Tokenized)
    Line()
    clog('Step 13: Saving dataframe, Sentences, Cleaned Sentences')    
    directory='datasets/'+args.dataset+'/'+args.task+'/'
    with open(directory + 'Sentences.json', 'w') as json_file:
        json.dump(Sentences_Tokenized, json_file)
    print('Saved Tokenized Sentences')
    print('_________________________________')
    with open(directory+'Cleaned_Sentences.json', 'w') as json_file:
        json.dump(CST, json_file)
    fdf_train.to_csv(directory+'dataset.csv',index=False)
    print('Saved Cleaned Tokenized Sentences')
    print('_________________________________')
    
main()
