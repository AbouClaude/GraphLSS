# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:07:47 2024

@author: hazem
"""

import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer



def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def Lemmatize_sentence(sentence):
    lemma_sentence=[]
    lemmatizer = WordNetLemmatizer()
    tokens= word_tokenize(sentence)
    
    for word in tokens:
        wntag = get_wordnet_pos(word)
        if wntag is None:# not supply tag in case of None
            lemma = lemmatizer.lemmatize(word)
            lemma_sentence.append(lemma)
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag) 
            lemma_sentence.append(lemma)

    return " ".join(lemma_sentence)


def compute_tfidf(article):
    #get tfidfvectorizer
    vectorizer = TfidfVectorizer()
    #fit into the article
    vectors = vectorizer.fit_transform(article)
    #convert the results to dataframe
    tf_idf = pd.DataFrame(vectors.todense())
    #get the name of the columns
    tf_idf.columns = vectorizer.get_feature_names_out()
    #transpose the dataframe
    tfidf_matrix = tf_idf.T
    #put the column numerical as the id of the sentences
    tfidf_matrix.columns = [str(i) for i in range(len(article))]
    #get sum of the value of the tfidf
    tfidf_matrix.insert(loc=0, column='sum_values', value=tfidf_matrix.sum(axis=1))
    #add the tagger as a column with its value to the dataframe
    tagger_values=[nltk.pos_tag([str(x)])[0][1] for x in tfidf_matrix.index]
    tfidf_matrix.insert(loc=0,column='Tagger',value=tagger_values)
    tfidf_matrix.insert(loc=2,column='repetition',value=np.count_nonzero(tfidf_matrix.iloc[:,2:],axis=1))
    #sort the dataframe by the sum_vales score and return the dataframe
    tfidf_matrix.sort_values(by='sum_values',ascending=False,inplace=True)
    return tfidf_matrix


def word_sentence_edges(df_tf):
    #source node which is the word start from 0
    s=0
    #word sentence edges article
    ws_edges=[]
    for index,row in df_tf.iterrows():
        #get the value>0 which means tfidf values exist
        row=row[row.values>0]
        #d destination sentence (source is words) distination is sentences
        Word_edges=[((s,int(d)),f) for d,f in zip(row.index,row.values)]
        s+=1
        ws_edges.append(Word_edges)
        
    return ws_edges

def Words_similarity(source_word,destination_word,word_embedding,word_index):
    
    ws=word_embedding.similarity(source_word,destination_word)
    
    #return ((source_word,destination_word),ws)
    if ws>0.6:
        return [(word_index[source_word],word_index[destination_word]),ws]
    else:
        return None