import numpy as np
def PrepareData(worddimy):
	count=0
	dictionary,good_sentence,bad_sentence,total_sen=set(),list(),list(),list()
	with open("movie_lines.txt",'r') as data_file:
		for sent in data_file:
			sen=sent.split(" +++$+++ ")
			sen=sen[len(sen)-1].split(" ")
			bad_char=["\n","<u>","<b>","<i>","</u>","</b>","</i>","\97",".","\"","\'","?","!",",","-",":","+","*","_","$","`",";"]
			for w in range(0,len(sen)):
				sen[w]=sen[w].lower()
				sen[w]=sen[w].replace("\ed","\'")
				for a in bad_char:
					sen[w]=sen[w].replace(a,"")
				sen[w]=sen[w].strip()
				dictionary.add(sen[w])
            	#print(sen[w])
			bsen=sen.copy()
			sen.append(1)
			good_sentence.append(sen)
			np.random.shuffle(bsen)
			bsen.append(-1)
			bad_sentence.append(bsen)
	total_sen=good_sentence+bad_sentence
	np.random.shuffle(total_sen)
	dictionary=list(dictionary)
	print(len(dictionary))
	with open("words.txt","w") as word_file:
		for x in range(0,len(dictionary)):
			word_file.write(dictionary[x]+"\n")
		word_file.close()
	wordvec=np.random.uniform(-2,2,(len(dictionary),5,worddimy))
	return dictionary,good_sentence,bad_sentence,wordvec,total_sen
