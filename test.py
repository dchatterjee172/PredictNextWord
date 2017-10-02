import tensorflow as tf
import numpy as np
import random as rn
dictionary=set()
good_sentence=list()
bad_sentence=list()
total_sen=list()
wordvec=0
worddimy=4
def PrepareData():
    count=0
    global dictionary,good_sentence,bad_sentence,wordvec,total_sen
    with open("movie_lines.txt",'r') as data_file:
        for sent in data_file:
            sen=sent.split(" +++$+++ ")
            sen=sen[len(sen)-1].split(" ")
            f=1
            for w in range(0,len(sen)):
            	sen[w]=sen[w].replace("\n","")
            	sen[w]=sen[w].replace(".","")
            	sen[w]=sen[w].replace("\"","")
            	sen[w]=sen[w].replace("?","")
            	sen[w]=sen[w].replace("!","")
            	sen[w]=sen[w].replace(",","")
            	sen[w]=sen[w].replace("-","")
            	sen[w]=sen[w].replace("\97","")
            	sen[w]=sen[w].replace(":","")
            	sen[w]=sen[w].replace("</u>","")
            	sen[w]=sen[w].replace("<u>","")
            	sen[w]=sen[w].replace("+","")
            	sen[w]=sen[w].replace("$","")
            	sen[w]=sen[w].lower()
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
    wordvec=np.random.uniform(-2,2,(len(dictionary),5,worddimy))
def WordtoVec():
	dt=tf.float32
	pre=tf.placeholder(name="pre",dtype=dt,shape=(1,1))
	coef=tf.placeholder(name="coef",dtype=dt,shape=(worddimy,1))
	actual=tf.placeholder(name="actual",dtype=dt,shape=(1,1))
	pregrad=tf.placeholder(name="stategradactual",dtype=dt,shape=(1,1))
	x=tf.get_variable("x1",shape=[5,1,worddimy])
	actx=tf.placeholder(name="actx",dtype=dt,shape=(5,1,worddimy))
	assignx=tf.assign(x,actx)
	state=(tf.matmul(tf.reduce_sum(x,0),coef)+pre)
	loss=tf.square((tf.nn.softsign(state)-actual))
	dxlast=tf.gradients(loss,x)
	dstatelast=tf.gradients(loss,state)
	dstate=pregrad
	dpre=tf.gradients(state,pre,grad_ys=pregrad)
	dx=tf.gradients(state,x,grad_ys=pregrad)
	with tf.Session() as sess:
		co=np.ones(shape=(worddimy,1))
		_i=0
		for words in total_sen:
			_i+=1
			#words=total_sen[2526]
			#print("Completed: ",_i/len(total_sen)*100)
			if len(words)<3:
				continue
			states=np.zeros(shape=(len(words)-1,1,1))
			res=np.array(words[len(words)-1]).reshape(1,1)
			_loss=0
			for w in range(0,len(words)-1):
				k=dictionary.index(words[w])
				x_=wordvec[k].reshape(5,1,worddimy)
				if w<len(words)-2:
					if w==0:
						inp={actx:x_,pre:[[0]],coef:co}
					else:
						inp={actx:x_,pre:states[w-1],coef:co}
					sess.run([assignx],feed_dict=inp)
					states[w]=sess.run(state,feed_dict=inp)
				else:
					inp={actx:x_,pre:states[w-1],coef:co,actual:res}
					z=sess.run([state,loss],feed_dict=inp)
					states[w]=z[0]
					print("loss: ",z[1][0][0])
					_loss=z[1][0][0]
			if(_loss<0.00000001):
				continue
			dpres=[[0]]
			for w in reversed(range(0,len(words)-1)):
				k=dictionary.index(words[w])
				x_=wordvec[k].reshape(5,1,worddimy)
				if w==len(words)-2:
					inp={actx:x_,pre:states[w-1],coef:co,actual:res}
					z=sess.run([dxlast,dstatelast],feed_dict=inp)
					wordvec[k]-=z[0][0].reshape(5,worddimy)*.1
					inp={actx:x_,pre:states[w-1],coef:co,actual:res,pregrad:z[1][0]}
					dpres=(sess.run(dpre,feed_dict=inp))[0]
				else:
					if w>0:
						inp={actx:x_,pre:states[w-1],coef:co,actual:res,pregrad:dpres}
						z=sess.run([dpre,dx],feed_dict=inp)
						dpres=z[0][0]
						wordvec[k]-=z[1][0].reshape(5,worddimy)*.1
					else:
						inp={actx:x_,pre:[[0]],coef:co,actual:res,pregrad:dpres}
						z=sess.run(dx,feed_dict=inp)
						wordvec[k]-=z[0].reshape(5,worddimy)*.1
PrepareData()
WordtoVec()
