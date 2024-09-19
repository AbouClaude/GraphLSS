# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:35:46 2024

@author: hazem
"""
import time
import pandas as pd
#NLTK
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from textblob import TextBlob, Word
from nltk.corpus import wordnet
import re

def Running_time(t, desc, p, k):
    elapsed_time = time.time() - t
    progress = round((p / k) * 100, 2)
    print(f"\033[2K\rRunning Time: {int(elapsed_time // 3600)} hours {int((elapsed_time // 60) % 60)} mins {int(elapsed_time % 60)} secs || Number of {desc}: {p}/{k} ({progress}%)", end="", flush=True)

#Extracting the columns values
#data: the loaded dataset, t= the type of dataset train validation or test
def Extracting_information(data,t):
    articles=[list(y.values())[0] for y in data[t]]
    summaries=[list(y.values())[1] for y in data[t]]
    return articles,summaries

def get_dataframe(articles,summaries):
    df=pd.DataFrame(data={'article':articles,
                                'summary':summaries},)
    return df

def form_set_df(s1,s2,s3,s4,set_names):
    all_elems = s1.union(s2).union(s3).union(s4)
    df = pd.DataFrame([[e in s1, e in s2,e in s3, e in s4] 
                   for e in all_elems], columns = set_names)
    return df.groupby(set_names).size()

def NLTK_Sentence_Tokenizer(df,feature):
    V=[]
    p=0
    t=time.time()
    for sample in df[feature]:
        V.append(sent_tokenize(sample))
        p+=1
        Running_time(t, 'Tokenize Sentences', p,len(df[feature])) if p%2000==0 else None
    Running_time(t, 'Tokenize Sentences', p,len(df[feature]))
    return V

def merge_ellipses(article):
    update_article=[]
    for sentence in article:
        if sentence not in ["...","» .","'.'",'"...','"','.','\\ .','ET).','!!"','* .','....']:
            update_article.append(sentence)
        else:
            try:
                update_article[-1] += '...'
            except:
                None
    return update_article

def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) #remove non-ascii
    string = re.sub(r",", " ", string) #remove comma ","
    string = re.sub(r"\. ", " ", string) #remove "."
    string = re.sub(r"\_", " ", string)  #remove "_"
    string = re.sub(r"!", " ", string)   #remove "!"
    string = re.sub(r"\(", " ", string)  #remove "("
    string = re.sub(r"\)", " ", string)  #remove ")"
    string = re.sub(r"\?", " ", string)  #remove "?"
    string = re.sub(r"\'", " ", string)  #remove "'"
    #other steps?
    #strip strings and convert them to lower
    string = string.strip().lower()
    #remove stop words
    #stopwords  related to english
    stop_words = stopwords.words('english')
    #tokenize the string
    text = word_tokenize(string)
    #remove stop word
    text = [word for word in text if word not in stop_words]
    #combine the token again
    return ' '.join(text)

def cleaned_tokenized_sentence(Sentences_Tokens):
    #start time
    t=time.time()
    all_cleaned_articles=[]
    k=len(Sentences_Tokens)
    p=0
    for article in Sentences_Tokens:
        cleaned_article=[clean_str(sentence) for sentence in article]
        all_cleaned_articles.append(cleaned_article)
        p+=1
        Running_time(t, 'Cleaning sentences articles', p,k) if p%50==0 else None
    Running_time(t, 'Cleaning sentences articles', p,k)
    return all_cleaned_articles
         
def check_short_sentences(CST,st):
    b=[len(x) for x in CST]
    print('Total number of sentences : ',sum(b))

    a=[len([a for a in x if len(a)<10 or " " not in a]) for x in CST]
    print('Number of short sentences of all articles : ',sum(a))
    

def merge_prev(article,cleaned_article):
    #updated article, updated cleaned article
    updated_a,updated_ca=[],[]
    #flag
    empty=False
    #iterate over cleaned article and article reversely
    for s,cs in zip(article,cleaned_article):
        #check if sentence less than 
        if (len(cs)<10 or " " not in cs) and len(updated_ca)!=0:
            #update cleaned article add the current short cleaned sentence to the previous 1 as (current + previous)
            updated_ca[-1]=updated_ca[-1]+" "+cs
            #update article add the current short sentence to the previous 1 as (current + previous)
            updated_a[-1]=updated_a[-1]+" "+s
        else:
            #add cleaned sentence to the cleaned article 
            updated_ca.append(cs)
            #add sentence to the article
            updated_a.append(s)
    #reverese the results of cleaned article and cleaned sentence and return them as results
    #updated_a=list(reversed(updated_a))
    #updated_ca=list(reversed(updated_ca))
    return updated_a,updated_ca


def Merge_last_word(article,cleaned_article):
    #if last sentence has no space
    if " " not in cleaned_article[-1] or len(cleaned_article[-1])<10:
        #merge the last sentence with the prev one
        article[-2]=article[-2]+" "+article[-1]
        cleaned_article[-2]=cleaned_article[-2]+" "+cleaned_article[-1]
        #____________________________________________
        #remove from both lists
        article.pop(-1)
        cleaned_article.pop(-1)