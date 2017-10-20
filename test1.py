import tensorflow as tf
import numpy as np
import data1
import signal
import tsne
good_sentence=list()
bad_sentence=list()
total_sen=list()
dictionary=list()
wordvec=0
worddimy=300
worddimx=1
hiddens=150
_loop=1
batch_mem=70
trainingpbatch=0
def start_tsne(file_name):
	global batch_mem,total_sen,wordvec,dictionary
	_dict=set()
	for x in range(0,batch_mem):
		for y in range(0,5):
			_dict.add(int(total_sen[x][y]))
	_dict=list(_dict)
	vec=np.ones(shape=(len(_dict),worddimy))
	for x in range(0,len(vec)):
		vec[x]=wordvec[_dict[x]].reshape(worddimy)
	y=tsne.tsne(vec, 2, 50, 20.0);
	with open(file_name,"w") as graph:
		for x in range(0,len(_dict)):
			graph.write(dictionary[_dict[x]]+" "+str(y[x][0])+" "+str(y[x][1])+"\n")
	graph.close()
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
	start_tsne("~wgraph0")
	dt=tf.float32
	pre=tf.placeholder(name="pre",dtype=dt,shape=(1,hiddens))
	coef=tf.placeholder(name="coef",dtype=dt,shape=(worddimx,worddimy,hiddens))
	coef1=tf.placeholder(name="coef",dtype=dt,shape=(hiddens,1))
	coef2=tf.placeholder(name="coef",dtype=dt,shape=(hiddens,hiddens))
	actual=tf.placeholder(name="actual",dtype=dt,shape=(1,1))
	pregrad=tf.placeholder(name="stategradactual",dtype=dt,shape=(1,hiddens))
	x=tf.get_variable("x1",shape=[worddimx,1,worddimy])
	actx=tf.placeholder(name="actx",dtype=dt,shape=(worddimx,1,worddimy))
	assignx=tf.assign(x,actx)
	rawstate=tf.zeros(shape=(1,hiddens),dtype=dt,name="rawstate")
	v=tf.constant(0)
	calrawstate=tf.while_loop(cond,body,[rawstate,x,v,coef])[0]
	state=(calrawstate+tf.matmul(pre,coef2))
	final=tf.nn.softsign(tf.matmul(state,coef1))
	#loss=tf.square(final-actual)
	loss=tf.losses.absolute_difference(actual,final)
	dxlast=tf.gradients(loss,x,name="lastxgrad")
	dstatelast=tf.gradients(loss,state,name="laststategrad")
	dcoeflast=tf.gradients(loss,coef,name="laststategrad")
	dpre=tf.gradients(state,pre,grad_ys=pregrad,name="pregrad")
	dx=tf.gradients(state,x,grad_ys=pregrad,name="xgrad")
	dcoef1=tf.gradients(loss,coef1,name="dcoef1")
	_graph=open("~graph","w")
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("tfg", sess.graph)
		co=np.random.uniform(-.2,.2,(worddimx,worddimy,hiddens))
		co1=np.random.uniform(-.15,.15,(hiddens,1))
		co2=np.random.uniform(-.15,.15,(hiddens,hiddens))
		for x in range(0,hiddens):
			for y in range (0, hiddens):
				if .5>np.random.uniform(0,1):
					co2[x][y]=0
		dco1=np.zeros(shape=(hiddens,1))
		_i=0
		_bloss=0
		_loss=0
		global batch_mem,trainingpbatch
		it=0
		c=0
		_trainingpbatch=trainingpbatch
		while it<len(total_sen) and _loop:
			#words=total_sen[2526]
			#print("Completed: ",_i/len(total_sen)*100)
			words=total_sen[it]
			it+=1
			_i+=1
			if _i==batch_mem:
				if _trainingpbatch>0:
					it-=_i
					_i=0
					_trainingpbatch-=1
				else:
					it=0
					_i=0
					_trainingpbatch=trainingpbatch
					print(it," loss ",_bloss/batch_mem)
					c+=1
					_graph.write(str(c)+" "+str(_bloss/batch_mem)+"\n")
				_bloss=0
			states=np.zeros(shape=(len(words)-1,1,hiddens))
			res=np.array(int(words[len(words)-1])).reshape(1,1)
			_out=0
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
					inp={actx:x_,pre:states[w-1],coef:co,actual:res,coef2:co2,coef1:co1}
					z=sess.run([state,loss,final],feed_dict=inp)
					#print(z[2][0],res)
					#print(z[0])
					states[w]=z[0]
					#_loss=z[1][0][0]
					_loss=z[1]
					_out=z[2][0][0]
			_bloss+=_loss
			#if(_loss<0.05):
			if((_out<0 and res[0][0]<0 and _loss<0.4)or(_out>0 and res[0][0]>0 and _loss<0.4)):
				continue
			dpres=[[0]]
			for w in reversed(range(0,len(words)-1)):
				k=int(words[w])
				x_=wordvec[k].reshape(worddimx,1,worddimy)
				if w==len(words)-2:
					inp={actx:x_,pre:states[w-1],coef:co,actual:res,coef2:co2,coef1:co1}
					z=sess.run([dxlast,dstatelast,dcoef1],feed_dict=inp)
					wordvec[k]=np.subtract(wordvec[k],z[0][0].reshape(worddimx,worddimy)*.15)
					#co1=np.subtract(co1,z[2][0]*.15)
					#print(z[0][0].reshape(worddimx,worddimy)*.5)
					inp={actx:x_,pre:states[w-1],coef:co,actual:res,pregrad:z[1][0],coef2:co2,coef1:co1}
					dpres=(sess.run(dpre,feed_dict=inp))[0]
				else:
					if w>0:
						inp={actx:x_,pre:states[w-1],coef:co,actual:res,pregrad:dpres,coef2:co2}
						z=sess.run([dpre,dx],feed_dict=inp)
						dpres=z[0][0]
						wordvec[k]=np.subtract(wordvec[k],z[1][0].reshape(worddimx,worddimy)*.15)
						#print(z[1][0].reshape(worddimx,worddimy)*.4)
					else:
						inp={actx:x_,pre:np.zeros(shape=(1,hiddens)),coef:co,actual:res,pregrad:dpres,coef2:co2}
						z=sess.run([dx],feed_dict=inp)
						wordvec[k]=np.subtract(wordvec[k],z[0][0].reshape(worddimx,worddimy)*.15)
						#print(z[0][0].reshape(worddimx,worddimy)*.4)
	_graph.close()
	start_tsne("~wgraph1")
signal.signal(signal.SIGINT, sig)
#good_sentence,bad_sentence,wordvec,total_sen=data1.PrepareData(worddimx,worddimy)
dictionary,total_sen,wordvec=data1.GetData(worddimx,worddimy)
WordtoVec()
