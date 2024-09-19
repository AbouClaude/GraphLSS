# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:39:00 2024

@author: hazem
"""

import time
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import itertools
import nltk
from itertools import combinations
import gensim.downloader as api
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tools.logger import *
import argparse


from Edge_Processing.Sentences_edges import *
from Edge_Processing.Words_edges import *

    




def Line():
    print("\033[1m========================================\033[0m")
    print('\n\n')
    
def clog(s):
    logger.info(s)
    print("________________________________________")

def Running_time(t, desc, p, k):
    elapsed_time = time.time() - t
    progress = round((p / k) * 100, 2)
    print(f"\033[2K\rRunning Time: {int(elapsed_time // 3600)} hours {int((elapsed_time // 60) % 60)} mins {int(elapsed_time % 60)} secs || Number of {desc}: {p}/{k} ({progress}%)", end="", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Preprocessing dataset')

    parser.add_argument('--dataset', type=str, default='Arxiv', help='The dataset directory.')
    parser.add_argument('--task', type=str, default='train', help='dataset [train|validation|test]')
    
    args = parser.parse_args()
    
    
    
    path='datasets/'+args.dataset+'/'+args.task+'/'
    '''
    clog("Step 1: Load Sentences and Adjecancy edges")
    df_graph=pd.DataFrame()
    with open(path+'Sentences.json', 'r') as file:
        df_graph['Sentences'] = json.load(file)
    with open(path+'sent_adj/'+'Edges.json', 'r') as file:
        df_graph['Sentences_edges'] = json.load(file)
    Line()
    clog("Step 2: Compute Sentences Edges")
    total_edge_adj=[]
    total_edge_sim=[]
    p=0
    t=time.time()
    Running_time(t,'Computing Sentences Edges',p,len(df_graph.Sentences))
    for sentences,edges in zip(df_graph.Sentences,df_graph.Sentences_edges):
        #get the embeddding
        edges=[tuple(x) for x in edges]
        embedding=Sentences_vector_embedding(sentences)
        #get the similarity of the embedding
        similarities=Sentences_similarity(embedding)
        #get the feature of adjecancy
        edge_adj=get_feature_adjecancy(edges,similarities)
        edge_sim=get_feature_similarity(similarities,edge_adj)
        total_edge_adj.append(edge_adj)
        total_edge_sim.append(edge_sim)
        p+=1
        Running_time(t,'Computing Sentences Edges',p,len(df_graph.Sentences)) if p%10==0 else None
           
    Running_time(t,'Computing Sentences Edges',p,len(df_graph.Sentences))
    print("")
    
    Line()
    clog("Step 3: Save Sentences Similarity edges and its feature and Sentence adjecancy feature")
    sentence_edges_features=[[x[1] for x in y] for y in total_edge_adj]
    
    #Edges adjecancy features
    with open(path+'sent_adj/'+'Features.json', 'w') as json_file:
        json.dump(sentence_edges_features, json_file)
    print('Saved adjecancy Features completed.')
    print("________________________________________")
    #sim edges
    similarity_edges=[[x[0] for x in y] for y in total_edge_sim]
    
    with open(path+'Sen_sim/'+'Edges.json', 'w') as json_file:  
        json.dump(similarity_edges, json_file)
    print('Saved Similarity Edges completed')
    print("________________________________________")
    #sim feaeture   
    similarity_features=[[x[1] for x in y] for y in total_edge_sim]
    
    with open(path+'Sen_sim/'+'Features.json', 'w') as json_file:
        json.dump(similarity_features, json_file)
    print('Saved Similarity features completed')
    print("________________________________________")
    Line()
    del df_graph
    
    clog('Step 4: Creating Words Sentences Edges')
    with open(path+'Cleaned_Sentences.json', 'r') as file:
        cleaned_tokenized_sentences = json.load(file)
    partofspeech=['JJ','JJR','JJS','NN','NNP','NNS','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']
    total_article_words,total_article_nodes=[],[]
    total_edges_ws,total_features_ws=[],[]
    t=time.time()
    p=0
    for article in cleaned_tokenized_sentences:
        #1. Lemmatize the sentence of the text
        lemm_article=[Lemmatize_sentence(x) for x in article]
    
    
    
        #2. Compute tf_idf lemmatized article
        df_tf=compute_tfidf(lemm_article)
    
        #3. Filter the desired part of speech
        df_tf=df_tf[df_tf.Tagger.isin(partofspeech)]
        
    
        #4. Filter the desired tf_idf sum score
        c=(df_tf.sum_values.mean()+df_tf.sum_values.median())/1.6
        df_tf=df_tf[df_tf.sum_values>c]
        
    
    
        #5. Filter uncessary words
       
        if 'et' in df_tf.index:
            df_tf.drop(['et'],axis=0,inplace=True)
        if 'al' in df_tf.index:
            df_tf.drop('al',axis=0,inplace=True)
        if 'fig' in df_tf.index:
            df_tf.drop('fig',axis=0,inplace=True)
        
    
    
        #6. Generating article words and append them to total words
        total_article_words.append([x for x in df_tf.index])
        #7. Generating article nodes and append to total nodes:
        total_article_nodes.append([x for x in range(len(df_tf.index))])
    
    
        #8. filter only the sentence columns
        df_tf=df_tf[df_tf.columns[3:]]
    
        #9. create ws edges
        ews=word_sentence_edges(df_tf)
        ews= list(itertools.chain(*ews))
    
    
        #10 append feature and edges to total
        total_edges_ws.append([x[0] for x in ews])
        total_features_ws.append([x[1] for x in ews])
        p+=1
        Running_time(t,'Words Edges',p,len(cleaned_tokenized_sentences)) if p%25==0 else None
    Running_time(t,'Words Edges',p,len(cleaned_tokenized_sentences))
    print("")
    Line()
    clog('Step 5: Saving the words tokens, words nodes, WS edges and WS features')
    with open(path+'Words_tokens.json', 'w') as json_file:
        json.dump(total_article_words, json_file)
    print('Saved words tokens completed.')
    print("________________________________________")
    
    with open(path+'Words_nodes.json', 'w') as json_file:
        json.dump(total_article_nodes, json_file)
    print('Saved words nodes completed.')
    print("________________________________________")
        
    with open(path+'WS_edges/'+'Edges.json', 'w') as json_file:
        json.dump(total_edges_ws, json_file)
    print('Saved WS edges completed.')
    print("________________________________________")
    with open(path+'WS_edges/'+'Features.json', 'w') as json_file:
        json.dump(total_features_ws, json_file)
    print('Saved WS features completed.')
    print("________________________________________")
    Line()
    del cleaned_tokenized_sentences
    '''
    clog('Step 6: Load Words tokens and Words Nodes')
    df_graph=pd.DataFrame()
    
    with open(path+'Words_tokens.json', 'r') as file:
        df_graph['Words'] = json.load(file)
    with open(path+'Words_nodes.json', 'r') as file:
        df_graph['Words_nodes'] = json.load(file)
       
    
    wv = api.load('word2vec-google-news-300')
    known_words=[[x for x in word_article if (x in wv.key_to_index.keys())] for word_article in df_graph.Words]
    total_dict_words=[{word: node for node,word in zip(nodes,words)} for nodes,words in zip(df_graph.Words_nodes,df_graph.Words)]
    
    
    t=time.time()
    p=0
    Line()
    clog('Step 7: WW Edges Computing')
    W_W_Edges=[]
    #iterate over all words and known words in the embedding
    Running_time(t,'Words words Edges',p,len(total_dict_words))
    for article_words,wx in zip(known_words,total_dict_words):
        wwe=[]
        #get combination of articles known words
        word_comb=list(combinations(article_words,2))
        #iterate over combination
        for comb in word_comb:
            #get the similarity of the two words
            a=Words_similarity(comb[0],comb[1],wv,wx)
            if a!=None:
                wwe.append(a)
        W_W_Edges.append(wwe)
        p+=1
        Running_time(t,'Words words Edges',p,len(total_dict_words)) if p%50==0 else None
    
    clog('Step 8: Save WW Edges WW features') 
    Running_time(t,'Words words Edges',p,len(total_dict_words))
    
    ww_edges=[[x[0] for x in y] for y in W_W_Edges]
    ww_features=[[float(x[1]) for x in y] for y in W_W_Edges]
    print("")
    with open(path+'WW_edges/'+'Edges.json', 'w') as json_file:
        json.dump(ww_edges, json_file)
    print('Saved Words Words edges completed.')
    print("________________________________________")
    with open(path+'WW_edges/'+'Features.json', 'w') as json_file:
        json.dump(ww_features, json_file)
    print('Saved Words Words features completed.')
    print("________________________________________")
    Line()
    
main()

        