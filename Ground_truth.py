# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:21:14 2024

@author: hazem
"""

import pandas as pd
import time

from tools.logger import *
import json
import argparse
import rouge_score
from rouge_score import rouge_scorer
def Frouge(hyps,refers):
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    v=scorer.score(hyps,refers)
    return v['rouge1'].fmeasure

#Get Sentences and Summary
def cal_label_Fscore(sentences,summary):
    
    #compute Rouge1 F1score for all sentences
    fscore1=pd.Series([round(Frouge(x,summary),4) for x in sentences])
    #sort the values of Fscore from the highest to the lowest with keeping the index (as sentence index)
    fscore1.sort_values(inplace=True,ascending=False)
    #max fscore equal to the first element
    max_fscore=fscore1[0]
    #
    labels=[fscore1.index[0]]
    for x in fscore1.index[1:]:
        #gather the selected sentences with the current sentences
        sent_gather=" ".join(pd.Series(sentences)[labels])+" "+sentences[x]
        #compute the Fscore of current sentence
        cur_fscore1=Frouge(sent_gather,summary)
        #if the current greater than max append the label index and modify the max
        if cur_fscore1>max_fscore:
            labels.append(int(x))
            #print(labels)
            max_fscore=cur_fscore1
    return labels

def Labeling_values(Sentences_list,Labeling_list):
    return [1 if i in Labeling_list else 0 for i in range(len(Sentences_list))]

def Running_time(t, desc, p, k):
    t = time.time() - t
    print("Running Time : "+str(int((t)//3600))," hours and ",
          str(int(((t)//60))%60)," mins and ",str(int((t)%60))
          ," secs","|| Number of ",desc," : " + str(p) + "/" + str(k),"   (",str(round((p/k)*100,2)),"%)     ", end="\r", flush=True)
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
    clog("Step 1: Load the articles and summaries dataframe")
    df_data=pd.read_csv(path+'dataset.csv',usecols=['summary'])
    print('dataframe Loaded successfully.')
    Line()
    clog("step 2: Load sentences json file and merge with dataframe")
    
    with open(path+'Sentences.json', 'r') as file:
        df_data['Sentences'] = json.load(file)
    print('Data Merged Successfully.')
    Line()
    clog("Step 3: Compute the labeling of the sentnences.")
    
    
    p=0
    t=time.time()
    labeling_list=[]
    k=len(df_data.Sentences)
    Running_time(t,'Labeling',p,len(df_data.Sentences)) if p%25==0 else None
    for sents,summary in zip(df_data.Sentences,df_data.summary):
        labels=cal_label_Fscore(sents, summary)
        
        labeling_list.append(labels)
        t2=time.time()
        Running_time(t,'Labeling',p,len(df_data.Sentences)) if p%25==0 else None
        p+=1
    Running_time(t,'Labeling',p,len(df_data.Sentences))
    print("Labeling Completed")
    Line()
    clog("Step 4: Get Labels values as e.g. [1,0,0,0,1,1 .. ].")
    
    label_values=[]
    for ls,s in zip(labeling_list,df_data.Sentences):
        label_values.append(Labeling_values(s,ls))
    del df_data
    Line()
    clog("Step 5: Save the label file.")
    
    with open(path+'Labels.json', 'w') as json_file:
        json.dump(label_values, json_file)
    print('Labels Saved Successfully')
    Line()
    
    
    

       
main()