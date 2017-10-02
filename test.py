import tensorflow as tf
import numpy as np
import random as rn
dictionary=set()
good_sentence=list()
bad_sentence=list()
wordvec=0
worddimy=4
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
                dictionary.add(w)
            good_sentence.append(sen)
            bad_sentence.append(sen)
            np.random.shuffle(bad_sentence[len(bad_sentence)-1])
    dictionary=list(dictionary)
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
	state=tf.nn.elu(tf.matmul(tf.reduce_sum(x,0),coef)+pre)
	loss=tf.square((tf.sigmoid(state)-actual))/2
	dxlast=tf.gradients(loss,x)
	dstatelast=tf.gradients(loss,state)
	dstate=pregrad
	dpre=tf.gradients(state,pre,grad_ys=pregrad)
	dx=tf.gradients(state,x,grad_ys=pregrad)
	with tf.Session() as sess:
		co=np.ones(shape=(worddimy,1))
		for words in bad_sentence:
			states=np.zeros(shape=(len(words),1,1))
			res=np.zeros(shape=(1,1))
			for w in range(0,len(words)):
				k=dictionary.index(words[w])
				x=wordvec[k].reshape(5,1,worddimy)
				if w<len(words)-1:
					if w==0:
						inp={actx:x,pre:[[0]],coef:co}
					else:
						inp={actx:x,pre:states[w-1],coef:co}
					sess.run([assignx],feed_dict=inp)
					states[w]=sess.run(state,feed_dict=inp)
				else:
					inp={actx:x,pre:states[w-1],coef:co,actual:res}
					z=sess.run([state,loss],feed_dict=inp)
					states[w]=z[0]
					print(z[1][0])
			dpres=[[0]]
			for w in reversed(range(0,len(words))):
				k=dictionary.index(words[w])
				x=wordvec[k].reshape(5,1,worddimy)
				if w==len(words)-1:
					inp={actx:x,pre:states[w-1],coef:co,actual:res}
					z=sess.run([dxlast,dstatelast],feed_dict=inp)
					wordvec[k]-=z[0][0].reshape(5,worddimy)*.05
					inp={actx:x,pre:states[w-1],coef:co,actual:res,pregrad:z[1][0]}
					dpres=(sess.run(dpre,feed_dict=inp))[0]
				else:
					if w>0:
						inp={actx:x,pre:states[w-1],coef:co,actual:res,pregrad:dpres}
						z=sess.run([dpre,dx],feed_dict=inp)
						dpres=z[0][0]
						wordvec[k]-=z[1][0].reshape(5,worddimy)*.05
					else:
						inp={actx:x,pre:[[0]],coef:co,actual:res,pregrad:dpres}
						z=sess.run(dx,feed_dict=inp)
						wordvec[k]-=z[0].reshape(5,worddimy)*.05
PrepareData()
WordtoVec()
