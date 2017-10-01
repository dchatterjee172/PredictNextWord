import tensorflow as tf
import numpy as np
import random as rn
english_words=set()
with open("wordsEn.txt") as word_file:
    english_words=set(word.strip().lower() for word in word_file)
