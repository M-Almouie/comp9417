import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=dir_path+'/'+sys.argv[1]
image_size=64
num_channels=3
images = []
files = os.listdir(image_path)
succs=0
for f in files:
	# Reading the image using OpenCV
	image = cv2.imread(image_path+'/'+f)
	# Resizing the image to our desired size and preprocessing will be done exactly as done during training
	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	images = []
	images.append(image)
	images = np.array(images, dtype=np.uint8)
	images = images.astype('float32')
	images = np.multiply(images, 1.0/255.0) 
	#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
	x_batch = images.reshape(1, image_size,image_size,num_channels)

	## Let us restore the saved model 
	sess = tf.Session()
	# Step-1: Recreate the network graph. At this step only graph is created.
	saver = tf.train.import_meta_graph('MURA-model.meta')
	# Step-2: Now let's load the weights saved using the restore method.
	saver.restore(sess, tf.train.latest_checkpoint('./'))

	# Accessing the default graph which we have restored
	graph = tf.get_default_graph()

	# Now, let's get hold of the op that we can be processed to get the output.
	# In the original network y_pred is the tensor that is the prediction of the network
	y_pred = graph.get_tensor_by_name("y_pred:0")

	## Let's feed the images to the input placeholders
	x= graph.get_tensor_by_name("x:0") 
	y_true = graph.get_tensor_by_name("y:0") 
	y_test_images = np.zeros((1, len(os.listdir('../../MURA-v1.1/train')))) 


	### Creating the feed_dict that is required to be fed to calculate y_pred 
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)
	# result is of this format [probabiliy_of_blind probability_of_normal]
	if result[0][0] > result[0][1]:
		type = "ABNORMAL"
		ans = result[0][0]
	else:
		type = "NORMAL" 
		ans = result[0][1]
	if type == "NORMAL" and "normal" in f:
		succs += 1
	if type == "BLIND" and "blind" in f:
		succs +=1
	print('For file named:\''+f+'\',Image is : '+type+', prediction certainty: '+str(ans)+'%')
print('Total successful prediction: '+str(succs/len(files))+'%')