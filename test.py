import tensorflow as tf
import numpy as np
import random as rn
dictionary=set()
def PrepareData():
    count=0
    with open("movie_lines.txt",'r') as data_file:
        for sentence in data_file:
            sen=sentence.split(" +++$+++ ")
            sen=sen[len(sen)-1].split(" ")
            f=1
            for w in sen:
                w.replace("\n","")
                w.replace(".","")
                w.replace("\"","")
                w.replace("?","")
                w.replace("!","")
                dictionary.update(w)
PrepareData()

