import tensorflow as tf
import numpy as np
import data
import data1
import signal
dictionary=set()
good_sentence=list()
bad_sentence=list()
total_sen=list()
wordvec=0
worddimy=10
worddimx=10
hiddens=6
_loop=1
def sig(signal,frame):
	global _loop
	_loop=0
def body(rawstate,x,v,coef):
	_r=rawstate+tf.matmul(x[v],coef[v])
	_v=v+1
	return [_r,x,_v,coef]
def cond(rawstate,x,v,coef):
	return v<worddimx
def WordtoVec():
	dt=tf.float32
	pre=tf.placeholder(name="pre",dtype=dt,shape=(1,hiddens))
	coef=tf.placeholder(name="coef",dtype=dt,shape=(worddimx,worddimy,hiddens))
	coef2=tf.placeholder(name="coef",dtype=dt,shape=(1,hiddens))
	actual=tf.placeholder(name="actual",dtype=dt,shape=(1,1))
	pregrad=tf.placeholder(name="stategradactual",dtype=dt,shape=(1,hiddens))
	x=tf.get_variable("x1",shape=[worddimx,1,worddimy])
	actx=tf.placeholder(name="actx",dtype=dt,shape=(worddimx,1,worddimy))
	assignx=tf.assign(x,actx)
	rawstate=tf.zeros(shape=(1,hiddens),dtype=dt,name="rawstate")
	v=tf.constant(0)
	calrawstate=tf.while_loop(cond,body,[rawstate,x,v,coef])[0]
	state=(calrawstate+tf.multiply(pre,coef2))
	final=tf.nn.softsign(tf.matmul(state,tf.ones(shape=(hiddens,1))))
	loss=tf.square(final-actual)
	dxlast=tf.gradients(loss,x,name="lastxgrad")
	dstatelast=tf.gradients(loss,state,name="laststategrad")
	dcoeflast=tf.gradients(loss,coef,name="laststategrad")
	dpre=tf.gradients(state,pre,grad_ys=pregrad,name="pregrad")
	dx=tf.gradients(state,x,grad_ys=pregrad,name="xgrad")
	dcoef=tf.gradients(state,coef,grad_ys=pregrad,name="xgrad")
	_graph=open("~graph","w")
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("tfg", sess.graph)
		co=np.random.uniform(-2,2,(worddimx,worddimy,hiddens))
		co2=np.random.uniform(0,1,(1,hiddens))
		_i=0
		_bloss=0
		_loss=0
		batch_mem=20
		trainingpbatch=0
		it=0
		c=0
		while it<len(total_sen) and _loop:
			#words=total_sen[2526]
			#print("Completed: ",_i/len(total_sen)*100)
			words=total_sen[it]
			it+=1
			_i+=1
			if _i==batch_mem:
				if trainingpbatch>0:
					it-=_i
					_i=0
					trainingpbatch-=1
				else:
					it=0
					_i=0
					trainingpbatch=0
					print(it," loss ",_bloss/batch_mem)
					c+=1
					_graph.write(str(c)+" "+str(_bloss/batch_mem)+"\n")
				_bloss=0
			states=np.zeros(shape=(len(words)-1,1,hiddens))
			res=np.array(int(words[len(words)-1])).reshape(1,1)
			for w in range(0,len(words)-1):
				k=int(words[w])
				#print(dictionary[int(words[w])])
				x_=wordvec[k].reshape(worddimx,1,worddimy)
				if w<len(words)-2:
					if w==0:
						inp={actx:x_,pre:np.zeros(shape=(1,hiddens)),coef:co,coef2:co2}
					else:
						inp={actx:x_,pre:states[w-1],coef:co,coef2:co2}
					sess.run([assignx],feed_dict=inp)
					states[w]=sess.run(state,feed_dict=inp)
				else:
					inp={actx:x_,pre:states[w-1],coef:co,actual:res,coef2:co2}
					z=sess.run([state,loss],feed_dict=inp)
					#print(z[0][0],res)
					states[w]=z[0]
					_loss=z[1][0][0]
			_bloss+=_loss
			if(_loss<0.00000001):
				continue
			dpres=[[0]]
			_dcoef=np.zeros(shape=(worddimx,worddimy,hiddens))
			for w in reversed(range(0,len(words)-1)):
				k=int(words[w])
				x_=wordvec[k].reshape(worddimx,1,worddimy)
				if w==len(words)-2:
					inp={actx:x_,pre:states[w-1],coef:co,actual:res,coef2:co2}
					z=sess.run([dxlast,dstatelast],feed_dict=inp)
					wordvec[k]-=z[0][0].reshape(worddimx,worddimy)*.01
					#print(z[0][0].reshape(worddimx,worddimy)*.5)
					#_dcoef+=z[2][0]
					inp={actx:x_,pre:states[w-1],coef:co,actual:res,pregrad:z[1][0],coef2:co2}
					dpres=(sess.run(dpre,feed_dict=inp))[0]
				else:
					if w>0:
						inp={actx:x_,pre:states[w-1],coef:co,actual:res,pregrad:dpres,coef2:co2}
						z=sess.run([dpre,dx],feed_dict=inp)
						dpres=z[0][0]
						#_dcoef+=z[2][0]
						wordvec[k]-=z[1][0].reshape(worddimx,worddimy)*.01
						#print(z[1][0].reshape(worddimx,worddimy)*.4)
					else:
						inp={actx:x_,pre:np.zeros(shape=(1,hiddens)),coef:co,actual:res,pregrad:dpres,coef2:co2}
						z=sess.run([dx],feed_dict=inp)
						wordvec[k]-=z[0][0].reshape(worddimx,worddimy)*.01
						#print(z[0][0].reshape(worddimx,worddimy)*.4)
						#_dcoef+=z[1][0]
				#co-=_dcoef*.1
	_graph.close()
signal.signal(signal.SIGINT, sig)
#dictionary,good_sentence,bad_sentence,wordvec,total_sen=data1.PrepareData(worddimx,worddimy)
dictionary,total_sen,wordvec=data1.GetData(worddimx,worddimy)
WordtoVec()
