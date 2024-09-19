# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:06:39 2024

@author: hazem
"""
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.metrics.scores import precision, recall, f_measure
import itertools
from rouge import Rouge

def rougef1_score(reference, hypothesis, ng):
    # Tokenize the text
    ref_tokens = word_tokenize(reference)
    hyp_tokens = word_tokenize(hypothesis)

    # Compute ROUGE-1 score
    ref_grams = set(ngrams(ref_tokens, ng))
    hyp_grams = set(ngrams(hyp_tokens, ng))
    f1 = f_measure(ref_grams, hyp_grams)
    if f1 is None:
        return 0
    else:
        return f1



      
def get_summary_batch(batch_index, highlight):
    bi = list(batch_index.cpu())
    h = list(highlight.iloc[bi])
    return " ".join([x for x in h])

def get_sentences_batch(batch_index, sentences):
    bi = list(batch_index.cpu())
    
    s = list(sentences.iloc[bi])
    s = list(itertools.chain(*s))
    return s

def mean_eval(e,r=5):
    res=round(np.mean(e),r)
    return res

        
def rouge_scores(reference, hypothesis, ng):
    # Tokenize the text
    ref_tokens = word_tokenize(reference)
    hyp_tokens = word_tokenize(hypothesis)

    # Compute n-grams
    ref_grams = set(ngrams(ref_tokens, ng))
    hyp_grams = set(ngrams(hyp_tokens, ng))

    # Compute precision, recall, and F1 score
    precision_score = precision(ref_grams, hyp_grams)
    recall_score = recall(ref_grams, hyp_grams)
    f1_score = f_measure(ref_grams, hyp_grams)

    return precision_score, recall_score, f1_score
    
def rougef1_L(hyp,ref):
    # Initialize Rouge
    rouge = Rouge()

    # Compute ROUGE scores
    scores = rouge.get_scores(hyp, ref)

    # Extract ROUGE-L score
    rouge_l_score = scores[0]["rouge-l"]["f"]

    return rouge_l_score

def summary_sentences(sents, pred):
    
    s = sents[pred == 1].tolist()
    s = " ".join(s)
    return s

'''

#GSR: Generated Summary Ratio, ASR: Actual Summary Ratio
#SRT: Summary ration Threshhold, CWT: Class weight threshhold
def check_Summary_ratio2(GSR,ASR,SRT=0.03,CWT=0.20):
    F=GSR-ASR
    print('Generated - Actual : ',F)
    if F>SRT*2:
        return CWT*2
    elif F>SRT:
        return CWT
    else:
        return 'Stop'
    
#SRT: Summary ration Threshhold, CWT: Class weight threshhold
#cw='Class weight
def check_Summary_ratio1(GSR,ASR,class_weight,epsillon=0.03):
    F=GSR-ASR
    cw=class_weight.tolist()
    r=np.log(cw[0]/cw[1])*(cw[0]/cw[1])
    
    print('Generated - Actual : ',F)
    if F>epsillon:
        return r-F*cw[0]
    else:
        return 'Stop'
 
    
def check_Summary_ratio3(GSR,ASR,epsillon=0.03):
    F=GSR-ASR
    
    r=-((F)+GSR)/np.log(F)+2*F
    print('Generated - Actual : ',F)
    if F>epsillon:
        return r
    else:
        return 'Stop'
'''

def check_Summary_ratio(GSR,ASR,epsillon=0.03):
    F=GSR-ASR
    
    r=-(GSR)/np.log(GSR)+GSR
    print(f'Generated - Actual : {GSR} - {ASR} = {F}')
    if F>epsillon:
        return r
    else:
        return 'Stop'
 
def check_Summary_ratio3(GSR,ASR,cw,epsillon=0.03):
    F=GSR-ASR
    factor=cw*(1-cw)/100
    r=-(GSR)/np.log(GSR)+GSR+cw*cw/150
    print(f'Generated - Actual : {GSR} - {ASR} = {F}')
    if F>epsillon:
        return r
    else:
        return 'Stop'
    
 