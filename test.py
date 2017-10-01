import tensorflow as tf
import numpy as np
import random as rn
dictionary=set()
good_sentence=list()
bad_sentence=list()
wordvec=0
def PrepareData():
    count=0
    global dictionary,good_sentence,bad_sentence,wordvec
    with open("movie_lines.txt",'r') as data_file:
        for sent in data_file:
            sen=sent.split(" +++$+++ ")
            sen=sen[len(sen)-1].split(" ")
            f=1
            for w in sen:
                w.replace("\n","")
                w.replace(".","")
                w.replace("\"","")
                w.replace("?","")
                w.replace("!","")
                dictionary.update(w)
            good_sentence.append(sen)
            bad_sentence.append(rn.shuffle(sen))
    dictionary=list(dictionary)
    wordvec=np.random.uniform(0,2,(len(dictionary),20))
PrepareData()
