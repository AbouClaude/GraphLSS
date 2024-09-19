# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:05:11 2024

@author: hazem
"""
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import itertools

def Sentences_vector_embedding(sentences,model=SentenceTransformer('all-MiniLM-L6-v2',device='cuda')):
    #get the embedding vector
    emb=model.encode(sentences,convert_to_tensor=False)
    return emb

def Sentences_similarity(embs,model=SentenceTransformer('all-MiniLM-L6-v2',device='cuda')):
    cosine_scores=util.cos_sim(embs,embs).tolist()
    return cosine_scores

def get_feature_adjecancy(edges,sim):
    #Sentence adjecancy feature
    edges_adj=[]
    
    for x in edges:
        #compute the similarity of existed edge
        feature=sim[x[0]][x[1]]
        
        edges_adj.append((x,feature))
    
    return edges_adj

#40% neighbor window with threshold over 0.7
def get_feature_similarity(sim,edges_adj,th=0.7):
    edges_sim=[]
    length_neighbor=(len(sim)*4)//10
    for x in range(len(sim)):
        sim_neighbors=pd.Series(sim[x][x+1:x+1+length_neighbor])
        sim_neighbors=sim_neighbors[sim_neighbors>th]
        
        if len(sim_neighbors)!=0:
            edges=[((x,y+x+1),score) for y,score in zip(sim_neighbors.index,sim_neighbors)]
            #print(edges)
            edges_sim.append(edges)
    edges_sim = list(itertools.chain(*edges_sim))
    if len(edges_sim)!=0:
        edges_sim= list(set(edges_sim) - set(edges_adj))
        edges_sim=sorted(edges_sim,key=lambda x: x[0])
    return edges_sim