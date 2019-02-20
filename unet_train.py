import tensorflow as tf
import numpy as np
import os
import cv2
import random
import convfunctions as conv #convolutionsal buildingblocks are in this module


#select which GPUs to use for training
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=["0","1","2"]


#training hypermparameters
BatchNum=32
InputDimension=[100,100,1]
GtDimensions=[100,100,1]
LearningRate=1e-3
NumIteration=1e6

#load the data from numpy
TrainInData=np.load('TrainInImages.npy')
TrainOutData=np.load('TrainOutImages.npy')
TestInData=np.load('TestInImages.npy')
TestOutData=np.load('TestOutImages.npy')

print("Image Preload Done")
NumTrainImages=TrainInData.shape[0]
print(NumTrainImages)

#create datasets
Train_x = tf.data.Dataset.from_tensor_slices(TrainInData)
Train_y = tf.data.Dataset.from_tensor_slices(TrainOutData)
#this script only uses train data, no test is included
#Test_x = tf.data.Dataset.from_tensor_slices(TestInData)
#Test_y = tf.data.Dataset.from_tensor_slices(TestOutData)
train_dataset = tf.data.Dataset.zip((Train_x,Train_y)).shuffle(500).repeat().batch(BatchNum)
#test_dataset = tf.data.Dataset.zip((Test_x,Test_y)).shuffle(500).repeat().batch(BatchNum)

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
testing_init_op = iterator.make_initializer(train_dataset)




CurrentInput =tf.cast(next_element[0],tf.float32)
InputGT=tf.cast(next_element[1],tf.float32)

#Data augmentation
#creating one array from input and output images
Combined=tf.concat([CurrentInput,InputGT],3)
print(Combined)
#random flip on all of them
Combined = tf.image.random_flip_left_right(Combined)
Combined = tf.image.random_flip_up_down(Combined)

#random rotation
Angles=tf.random_uniform([BatchNum],-0.1, 0.1)
Combined=tf.contrib.image.rotate(Combined,Angles)
#cutting out the middle part
Combined=Combined[:,15:135 ,15:135 ,:]

#random crop
Combined=tf.random_crop(Combined,  size = [BatchNum,InputDimension[0],InputDimension[1],2])


CurrentInput=tf.expand_dims(Combined[:,:,:,0],-1)
InputGT=tf.expand_dims(Combined[:,:,:,1],-1)


#CurrentInput= tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), CurrentInput)
#InputGT= tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), Input	GT)


AugemntedIn=CurrentInput
AugemntedOut=InputGT

#convolutional netwrok, number of kernels per layer
NumKernels=[InputDimension[2],16,32,64,128]
ImgSizes=[]
LayerLefts=[]
LayerNum=0
#creating the downscaling part of the net
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

#creating the upscaling part of the net
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
    L2 =tf.square(tf.subtract(InputGT,Enhanced))
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
    L2Loss =tf.reduce_mean( L2)


with tf.name_scope('optimizer'):	
    #Use ADAM optimizer
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(SoftL1Loss )

print("Computation graph building Done")

Init = tf.global_variables_initializer()

with tf.Session() as Sess:
	Sess.run(Init)
	Saver = tf.train.Saver()
	Sess.run(training_init_op)  
	Step = 1
	while Step < NumIteration:
		#execute teh session
		_,L,InputImages, GtImages,OutputImages = Sess.run([Optimizer, SoftL1Loss ,AugemntedIn,AugemntedOut,Enhanced])
		#print loss and accuracy at every 10th iteration
		if (Step%10)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Loss:" + str(L))
		#save samples
		if (Step%1000)==0:
			for i in range(3):
				cv2.imwrite('samples/'+str(Step).zfill(5)+'_'+str(i)+'_gt.png',((GtImages[i,:,:])*255.0))
				cv2.imwrite('samples/'+str(Step).zfill(5)+'_'+str(i)+'_in.png',((InputImages[i,:,:,:])*255.0))
				cv2.imwrite('samples/'+str(Step).zfill(5)+'_'+str(i)+'_out.png',((OutputImages[i,:,:,:])*255.0))
                #save checkpoint
		if (Step%10000)==0:
			print('Saving model...')
			print(Saver.save(Sess, "./checkpoint/"))
		Step+=1
