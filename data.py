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
	with open("movie_lines.txt",'r') as data_file:
		for sent in data_file:
			sen=sent.split(" +++$+++ ")
			sen=sen[len(sen)-1].split(" ")
			if len(sen)<4:
				continue
			bad_char=["\n","<u>","<html>","</html>","<pre>","</pre>","\t","=","<b>","<i>","</u>","</b>","</i>","\97",".","\"","\'","?","!",",","-",":","+","*","_","$","`",";","[","]","&","<",">","~","|","{","}","(",")"]
			for w in range(0,len(sen)):
				sen[w]=sen[w].lower()
				sen[w]=sen[w].replace("\ed","\'")
				for a in bad_char:
					sen[w]=sen[w].replace(a,"")
				sen[w]=sen[w].strip()
				dictionary.add(sen[w])
            	#print(sen[w])
			bsen=sen.copy()
			sen.append("1")
			good_sentence.append(sen)
			np.random.shuffle(bsen)
			bsen.append("-1")
			bad_sentence.append(bsen)
	total_sen=good_sentence+bad_sentence
	np.random.shuffle(total_sen)
	dictionary=list(dictionary)
	dictionary.sort()
	print(len(dictionary))
	with open("words.txt","w") as word_file:
		for x in range(0,len(dictionary)):
			word_file.write(dictionary[x]+" ")
		word_file.close()
	with open ("sentence.txt","w") as sen_file:
		for words in total_sen:
			for x in range(0,len(words)-1):
				words[x]=str(binsearch(dictionary,words[x]))
			sen_file.write(" ".join(words)+"\n")
		sen_file.close()	
	wordvec=np.random.uniform(-2,2,(len(dictionary),5,worddimy))
	return dictionary,good_sentence,bad_sentence,wordvec,total_sen
def GetData(worddimx,worddimy):
	dictionary,total_sen=list(),list()
	with open("words.txt","r") as word_file:
		x=word_file.read()
		x=x.replace("\n","")
		dictionary=x.split(" ")
		word_file.close()
	with open("sentence.txt","r") as sen_file:
		for words in sen_file:
			words=words.replace("\n","")
			words=words.split(" ")
			total_sen.append(words)
		sen_file.close()
	wordvec=np.random.uniform(-2,2,(len(dictionary),worddimx,worddimy))
	return dictionary,total_sen,wordvec
