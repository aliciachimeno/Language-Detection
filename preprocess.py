import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.lm import Vocabulary
import numpy as np
import pandas as pd


#Tokenizer function. You can add here different preprocesses.
def preprocess(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or lemmatization.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    
    for snt,ind in zip(sentence.values,sentence.index):
        if (len(snt.split(" ")) < 10):      #Potential Chinese, Korean, Indu and Japanese 
            #pr_snt = removeInfobtPar(snt)
            #pr_snt = removePunctuation(pr_snt)
            pr_snt = sntSplit(snt)
        else:                               #Other languages
            #pr_snt = removeInfobtPar(snt)
            pr_snt = removePunctuation(snt)
        sentence.loc[ind] = pr_snt
    
    return sentence,labels

def removePunctuation(snt):

    #Tokenization (remove punctuation)
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(snt)

    #Reconstruction
    snt = ' '.join(words)

    return snt

def removeInfobtPar(snt):

    #Tokenization (remove words between parenthesis)
    tokenizer = RegexpTokenizer(r'\([^)]*\)|\s', gaps=True)
    words = tokenizer.tokenize(snt)
    
    #Reconstruction
    snt = ' '.join(words)
    
    #Tokenization (remove words between —)
    tokenizer = RegexpTokenizer(r'\—[^)]*\—|\s', gaps=True)
    words = tokenizer.tokenize(snt)
    
    #Reconstruction
    snt = ' '.join(words)
    
    #Tokenization (remove words between quotation marks)
    tokenizer = RegexpTokenizer(r'\"[^)]*\"|\s', gaps=True)
    words = tokenizer.tokenize(snt)

    #Reconstruction
    snt = ' '.join(words)

    #Tokenization (remove words between »)
    tokenizer = RegexpTokenizer(r'\«[^)]*\»|\s', gaps=True)
    words = tokenizer.tokenize(snt)

    #Reconstruction
    snt = ' '.join(words)

    return snt

def sntSplit(snt):

    #Put spaces every 2 characters
    n = 2
    snt = ' '.join(a + b for a,b in zip(snt[::n], snt[1::n]))

    return snt


#Future Work: Classify languages for each root to further specialize preprocessing
def createVocabulary (languages):
    
    doc = pd.read_csv("dataset.csv")
    corpus = ''
    for language in languages:
        text = doc[doc["language"] == language]["Text"]
        for sentence in text.values:
            corpus = corpus + sentence

    vocab = Vocabulary(corpus, unk_cutoff=int(len(corpus)/300))

    return vocab
    
