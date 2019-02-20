import tensorflow as tf
import numpy as np
import os
import cv2
import random
import convfunctions as conv

#seelct GPUS
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#test parameters, single batch
BatchNum=1
InputDimension=[100,100,1]
GtDimensions=[100,100,1]
LearningRate=1e-4
NumIteration=1e6


print("Image Preload Done")
#Number of images is ~56k

InputData = tf.placeholder(tf.float32, [BatchNum]+InputDimension) #network input
InputGT = tf.placeholder(tf.float32, [BatchNum]+GtDimensions) #expected network output

CurrentInput=InputData
NumKernels=[InputDimension[2],64,128,256,512]
ImgSizes=[]
LayerLefts=[]
LayerNum=0
for n in range(0,len(NumKernels)-2):
   with tf.variable_scope('conv'+str(LayerNum)):
      LayerLeft,LayerNum = conv.ConvBlock(CurrentInput ,1,[3,3,NumKernels[n],NumKernels[n+1]],LayerNum)
      W = tf.get_variable('Wdown',[3, 3, NumKernels[n+1],NumKernels[n+1]])
      #CurrentInput=tf.nn.conv2d(LayerLeft,W,strides=[1,2,2,1],padding='SAME') #downsampling with strided conv
      CurrentInput = tf.nn.max_pool(LayerLeft , ksize=[1,2, 2, 1], strides=[ 1, 2, 2, 1], padding='SAME') #downsampling with pooling
      print(CurrentInput)
      LayerLefts.append(LayerLeft)
      ImgSizes.append([int(LayerLeft.get_shape()[1]),int(LayerLeft.get_shape()[2])])
      LayerNum +=1
CurrentInput,LayerNum = conv.ConvBlock(CurrentInput ,2,[3,3,NumKernels[-2],NumKernels[-1]],LayerNum)

print(CurrentInput)
for n in range(len(NumKernels)-1,1,-1):
	with tf.variable_scope('conv'+str(LayerNum)):
		W = tf.get_variable('W',[3, 3, NumKernels[n-1], NumKernels[n]])
		LayerRight=tf.nn.conv2d_transpose(CurrentInput, W,  [BatchNum, ImgSizes[n-2][0], ImgSizes[n-2][1], NumKernels[n-1]], [1, 2, 2 , 1], padding='SAME', name=None)
		print(LayerRight)
		Bias  = tf.get_variable('B',[NumKernels[n-1]])
		LayerRight=tf.add(LayerRight,Bias )
		LayerRight= conv.LeakyReLU(LayerRight)
		LayerNum +=1
		CurrentInput=tf.concat([LayerRight,LayerLefts[n-2] ],3)
		CurrentInput,LayerNum= conv.ConvBlock(CurrentInput ,2,[3,3,NumKernels[n],NumKernels[n-1]],LayerNum)

with tf.variable_scope('conv'+str(LayerNum)):
	W = tf.get_variable('W',[3, 3, NumKernels[1], GtDimensions[-1]])
	LayerOut= tf.nn.conv2d(CurrentInput,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
	Bias  = tf.get_variable('B',[GtDimensions[-1]])
	LayerOut= tf.add(LayerOut, Bias)
#no nonlinearity at the end
#LayerOut= LeakyReLU(LayerOut)
#Enhanced=tf.nn.sigmoid(LayerOut)
Enhanced=LayerOut
print(Enhanced)
print(InputGT)
# Define loss and optimizer
with tf.name_scope('loss'):
    # L1 loss
    AbsDif=tf.abs(tf.subtract(InputGT,Enhanced))

    #this part implements soft L1
    Comp = tf.constant(np.ones(AbsDif.shape), dtype = tf.float32)
    SmallerThanOne = tf.cast(tf.greater(Comp, AbsDif),tf.float32)
    LargerThanOne= tf.cast(tf.greater(AbsDif, Comp ),tf.float32)   
    ValuestoKeep=tf.subtract(AbsDif, tf.multiply(SmallerThanOne ,AbsDif))
    ValuestoSquare=tf.subtract(AbsDif, tf.multiply(LargerThanOne,AbsDif))
    SoftL1= tf.add(ValuestoKeep, tf.square(ValuestoSquare)) 
 
    #average loss
    SoftL1Loss = tf.reduce_mean( SoftL1)
    L1Loss = tf.reduce_mean( AbsDif)
    


with tf.name_scope('optimizer'):	
    #Use ADAM optimizer this is currently the best performing training algorithm in most cases
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(SoftL1Loss )

print("Computation graph building Done")

Init = tf.global_variables_initializer()


with tf.Session() as Sess:
	Sess.run(Init)
	Saver = tf.train.Saver()
	#Saver.restore(Sess, "./3/")
        #restore the save network
	Saver.restore(Sess, "./checkpoint_test/")
	Step = 1
	
	TrainInPath='images/test3/im200'
	ImgInd=0
	for file in os.listdir(TrainInPath):
		print(file)
		fpart=file.split('.')[0]
		parts=file.split('_')
		YStart=int(parts[5])-1
		YEnd=int(parts[6])
		XStart=int(parts[8])-1
		XEnd=int(parts[9].split(".")[0])
		if os.path.isfile(os.path.join(TrainInPath, file)):
			origimg=cv2.imread(os.path.join(TrainInPath, file))
			img=origimg[:100,:100,:]
			gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
			Img=np.reshape(gray,[1,100,100,1])
	
			#run the netwrok on the test images - cut out four different part - these calls could be together in batches of four
			OutputImages = Sess.run([Enhanced], feed_dict={InputData: Img})
			OutputImages=OutputImages[0]
			print(OutputImages.shape)
			temp = (OutputImages[0,:,:,0])
			cv2.imwrite('out/'+"exp1_im_1_crop_I_"+str(XStart)+"_"+str(XEnd-50)+"_J_"+str(YStart)+"_"+str(YEnd-50)+'.png',(temp)*255.0)
			img=origimg[50:150,:100,:]
			gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
			Img=np.reshape(gray,[1,100,100,1])
	
			OutputImages = Sess.run([Enhanced], feed_dict={InputData: Img})
			OutputImages=OutputImages[0]
			print(OutputImages.shape)
			temp = (OutputImages[0,:,:,0])
			cv2.imwrite('out/'+"exp1_im_1_crop_I_"+str(XStart)+"_"+str(XEnd-50)+"_J_"+str(YStart+50)+"_"+str(YEnd)+'.png',(temp)*255.0)
			img=origimg[:100,50:150,:]
			gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
			Img=np.reshape(gray,[1,100,100,1])
	
			OutputImages = Sess.run([Enhanced], feed_dict={InputData: Img})
			OutputImages=OutputImages[0]
			print(OutputImages.shape)
			temp = (OutputImages[0,:,:,0])
			cv2.imwrite('out/'+"exp1_im_1_crop_I_"+str(XStart+50)+"_"+str(XEnd)+"_J_"+str(YStart)+"_"+str(YEnd-50)+'.png',(temp)*255.0)
			img=origimg[50:150,50:150,:]
			gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
			Img=np.reshape(gray,[1,100,100,1])
	
			OutputImages = Sess.run([Enhanced], feed_dict={InputData: Img})
			OutputImages=OutputImages[0]
			print(OutputImages.shape)
			temp = (OutputImages[0,:,:,0])
			cv2.imwrite('out/'+"exp1_im_1_crop_I_"+str(XStart+50)+"_"+str(XEnd)+"_J_"+str(YStart+50)+"_"+str(YEnd)+'.png',(temp)*255.0)




