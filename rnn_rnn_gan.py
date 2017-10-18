import argparse
import matplotlib.pyplot as plt
import math
import json
import scipy
import scipy.io
import os
import sys
import copy, numpy as np
from PIL import Image
import glob
import matplotlib
from matplotlib import cm  # Colormaps
from matplotlib.colors import LogNorm  # Log colormaps
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Merge
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import SimpleRNN,LSTM
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist

def generator_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=False, input_shape=(3, 100*100)))
    model.add(Dense(100*100))
    model.add(Reshape((1, 100*100), input_shape=(100*100,)))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy');
    return model


def discriminator_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=False, input_shape=(4, 100*100)))
    model.add(Dense(100*100))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy');
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    list_of_dicriminator_inputs = []

    for _ in range(3):
        auxiliary_model = Sequential()
        auxiliary_model.add(Reshape((1, 100*100), input_shape=(100 * 100,)))
        list_of_dicriminator_inputs.append(auxiliary_model)
    list_of_dicriminator_inputs.append(generator)

    extended_generator_output = Merge(list_of_dicriminator_inputs,
                                      mode="concat",
                                      concat_axis=1)
    model.add(extended_generator_output)
    discriminator.trainable = False
    model.add(discriminator)
    #print ("MODEL SUMMARY:")	
    #print model.summary()
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image


def train(BATCH_SIZE):
	np.random.seed(seed=1)
	image_list = []
	datasetname='/path/to/data/set'
	dirs=sorted(next(os.walk(datasetname))[1])
	leng=len(dirs)
	width=100;
	height=100;
	inps=width*height
	label=[];
	train_set=[];
	test_set=[];
	fgnet=[];
	prs_id=0;
	maxlen = 3
	
	for j in range(leng):
	    #print(j)
	    dir_name=datasetname+'/'+dirs[j]+'/*.png'
		#print(dir_name)
            for filename in sorted(glob.glob(dir_name)):
	        #print(filename)
                im=Image.open(filename)
                im=im.resize((height,width),Image.ANTIALIAS)
                im=im.getdata();
                im=im.convert('L')
                im =np.array(im)
                im=im.ravel();
    	        fgnet.append(im);
                label.append(j+1);
	print len(fgnet)
	fgnet=np.array(fgnet); #fgnet dataset, holds the images of fgnet
	label=np.array(label); #label store the label of the image


	X = np.zeros((fgnet.shape[0], maxlen, height*width))
	y = np.zeros((fgnet.shape[0], height*width))

	x_idx1_cnt=0;
	x_idx2_cnt=0;
	y_idx1_cnt=0;
	for i in range(0,fgnet.shape[0]):
		if(x_idx2_cnt<maxlen):
		    X[x_idx1_cnt,x_idx2_cnt,:]=fgnet[i, :];
		    x_idx2_cnt=x_idx2_cnt+1;
    		else:
    	            y[y_idx1_cnt,:]=fgnet[i,:];
    	    	    x_idx2_cnt=0;
    	    	    y_idx1_cnt=y_idx1_cnt+1;
    	    	    x_idx1_cnt=x_idx1_cnt+1;

 
	X=X[0:x_idx1_cnt, :, :];
	y=y[0:y_idx1_cnt, :];

	X = X.astype('float32')/255;
	y = y.astype('float32')/255;

	train_len=2000
	test_len=526
        print X.shape[0]
	indices = np.random.permutation(X.shape[0])
	training_idx, test_idx = indices[:train_len], indices[train_len:]
	print(training_idx)
	print('-'*50)
	print(test_idx)

	trainX, testX = X[training_idx,:,:], X[test_idx,:,:];
	trainy,testy=y[training_idx,:],y[test_idx,:];

	np.save('indices',indices);
	np.save('trainX',trainX);
	np.save('testX',testX);
	np.save('trainy',trainy);
	np.save('testy',testy);
	
	trainX=np.load('trainX.npy');
	testX=np.load('testX.npy');
	trainy=np.load('trainy.npy');
	testy=np.load('testy.npy');

	print(trainX.shape)
	print(testX.shape)
	print(trainy.shape)
	print(testy.shape)

	print ("trainX shape is")
	print (trainX.shape)
	print ("trainy shape is")
	print (trainy.shape)
    
	#(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train=trainX
	y_train=trainy
	X_test=testX
	y_test=testy
	print X_train.shape,y_train.shape
	X_train = (X_train.astype(np.float32) - 127.5)/127.5
	X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
        print ("X_train shape is")
	print (X_train.shape)
	discriminator = discriminator_model()
	generator = generator_model()
	discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
	d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)  #lr=0.0005
	g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)  #lr=0.0005
	generator.compile(loss='binary_crossentropy', optimizer="SGD")
	discriminator_on_generator.compile(
	    loss='binary_crossentropy', optimizer=g_optim)
	discriminator.trainable = True
	discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
	noise = np.zeros((BATCH_SIZE, 3, 100*100))
	target = np.zeros((BATCH_SIZE, 1, 100, 100))
	for epoch in range(100): 
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
            for index in range(int(X_train.shape[0]/BATCH_SIZE)):
                for i in range(BATCH_SIZE):
                    noise[i, :] = X_train[index*BATCH_SIZE+i]
		    target[i, :] = y_train[index*BATCH_SIZE+i].reshape(1,100,100)
                image_batch = y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                generated_images = generator.predict(noise, verbose=0)
		generated_images = generated_images.reshape(BATCH_SIZE,1,100,100)
                if index % 20 == 0:
                    image = combine_images(generated_images)
                    image = image*127.5+127.5
                    Image.fromarray(image.astype(np.uint8)).save(
                        str(epoch)+"_"+str(index)+".png")
		image_batch = image_batch.reshape(BATCH_SIZE,1,100,100)
		#print image_batch.shape 	
		#print generated_images.shape
		X=np.zeros((BATCH_SIZE, 4, 100*100))
		temp=np.zeros((BATCH_SIZE, 4, 100*100))
		print noise[i].shape,generated_images[i].shape,X.shape
		for i in range(BATCH_SIZE):
                    X[i] = np.concatenate((noise[i],generated_images[i].reshape(1,10000)))
		for i in range(BATCH_SIZE):
		    temp1 = np.concatenate((noise[i],target[i].reshape(1,10000)))
		    temp1=temp1.reshape(1,4,10000)
		    temp[i]=temp1
		y = [0] * BATCH_SIZE + [1] * BATCH_SIZE
                d_loss = discriminator.train_on_batch(np.concatenate((X,temp)),y)
                print("batch %d d_loss : %f" % (index, d_loss))
                for i in range(BATCH_SIZE):
                    noise[i, :] = X_train[i]
                discriminator.trainable = False
                g_loss = discriminator_on_generator.train_on_batch(
                    [noise[:,0,:],noise[:,1,:],noise[:,2,:],noise], [1] * BATCH_SIZE)
                discriminator.trainable = True
                print("batch %d g_loss : %f" % (index, g_loss))
                if index % 499 == 1:
                    generator.save_weights('generator'+str(epoch)+'_'+str(index), True)
                    discriminator.save_weights('discriminator'+str(epoch)+'_'+str(index), True)


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator38_500')
    trainX=np.load('trainX.npy');
    testX=np.load('testX.npy');
    trainy=np.load('trainy.npy');
    testy=np.load('testy.npy');
    print ("testX shape is")
    print (testX.shape)
    print ("testy shape is")
    print (testy.shape)
    
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train=trainX
    y_train=trainy
    X_test=testX
    y_test=testy
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
	index=1
        noise = np.zeros((BATCH_SIZE,3,100*100))
        for i in range(BATCH_SIZE):
            noise[i, :] = X_train[index*BATCH_SIZE+i]
        generated_images = generator.predict(noise, verbose=1)
	generated_images = generated_images.reshape(BATCH_SIZE,1,100,100)
	print generated_images.shape
        image = combine_images(generated_images)
    image = image*127.5+127.5
    print("Almost done\n")
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image_on_train_set.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
