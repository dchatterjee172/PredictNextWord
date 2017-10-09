""" I am creating this new data genaration script for the dataset i got from https://www.ngrams.info/
As i cannot push their dataset, only this file will be there"""
import numpy as np
def binsearch(words,d):
	a,b=0,len(words)-1
	mid=int((a+b)/2)
	while(a<=b):
		if words[mid]==d:
			return mid
		elif words[mid]<d:
			a=mid+1
		else:
			b=mid-1
		mid=int((a+b)/2)	
	return -1
def PrepareData(worddimx,worddimy):
	dictionary,good_sentence,bad_sentence,total_sen=set(),list(),list(),list()
	with open("~w5_.txt",'r') as data_file:#this file is not published
		for sen in data_file:
			sen=sen.replace("\n","")
			sen=sen.split("\t")
			sen.pop(0)
			for w in range(0,len(sen)):
				sen[w]=sen[w].lower()
				sen[w]=sen[w].strip()
				dictionary.add(sen[w])
			bsen=sen.copy()
			sen.append("1")
			good_sentence.append(sen)
			#np.random.shuffle(bsen)
			bsen.append("-1")
			bad_sentence.append(bsen)
	#total_sen=good_sentence+bad_sentence
	#np.random.shuffle(total_sen)
	dictionary=list(dictionary)
	dictionary.sort()
	for sen in bad_sentence:
		a=np.random.randint(0,5)
		b=np.random.randint(0,len(dictionary))
		sen[a]=dictionary[b]
	a,b=0,0
	while b<len(good_sentence):
		if(a==b):
			total_sen.append(good_sentence[a])
			a=a+1
		else:
			total_sen.append(bad_sentence[b])
			b=b+1
	#total_sen=good_sentence+bad_sentence
	#np.random.shuffle(total_sen)
	print(len(dictionary))
	with open("~words.txt","w") as word_file:
		for x in range(0,len(dictionary)):
			word_file.write(dictionary[x]+" ")
		word_file.close()
	with open ("~sentence.txt","w") as sen_file:
		for words in total_sen:
			for x in range(0,len(words)-1):
				words[x]=str(binsearch(dictionary,words[x]))
			sen_file.write(" ".join(words)+"\n")
		sen_file.close()	
	wordvec=np.random.uniform(-2,2,(len(dictionary),5,worddimy))
	return dictionary,good_sentence,bad_sentence,wordvec,total_sen
def GetData(worddimx,worddimy):
	dictionary,total_sen=list(),list()
	with open("~words.txt","r") as word_file:
		x=word_file.read()
		x=x.replace("\n","")
		dictionary=x.split(" ")
		word_file.close()
	with open("~sentence.txt","r") as sen_file:
		for words in sen_file:
			words=words.replace("\n","")
			words=words.split(" ")
			total_sen.append(words)
		sen_file.close()
	#np.random.shuffle(total_sen)
	wordvec=np.random.uniform(-1,1,(len(dictionary),worddimx,worddimy))
	return dictionary,total_sen,wordvec
					
