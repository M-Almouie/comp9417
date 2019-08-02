import math, random, os, sys, time, glob, cv2
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

# Hyperparameters
batchSize = 16
imageSize = 64
numChannels = 3
categories = ['positive', 'negative']

def createPlaceholders():
	x = tf.placeholder(tf.float32, shape=[None, imageSize, imageSize, numChannels], name='x')
	y = tf.placeholder(tf.float32, shape=[None, len(categories)], name='y')
	yClasses = tf.argmax(y, dimension=1)
	return x, y, yClasses

# Network parameters
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

filter_size_conv4 = 3
num_filters_conv4 = 32

filter_size_conv5 = 3
num_filters_conv5 = 64

fc_layer_size = 128

# Tensorboard placeholders
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def preprocessImages(trainingPath):
  images = []
  labels = [] 
  img_names = []
  print('Processing training images:------------------')
  '''for bp in bodyParts:
    print("----------------- Reading Body Part: " + bp)
    patients = os.listdir(trainingPath + "/" + bp)'''
  for fields in categories:   
    index = categories.index(fields)
    print('Now going to read {} files (Index: {})'.format(fields, index))
    path = os.path.join(trainingPath, fields, '*g')
    files = glob.glob(path)
    for fl in files:
      image = cv2.imread(fl)
      image = cv2.resize(image, (imageSize, imageSize),0,0, cv2.INTER_LINEAR)
      image = image.astype(np.float32)
      image = np.multiply(image, 1.0 / 255.0)
      images.append(image)
      label = np.zeros(len(categories))
      label[index] = 1.0
      labels.append(label)
      flbase = os.path.basename(fl)
      img_names.append(flbase)
  images = np.array(images)
  labels = np.array(labels)
  img_names = np.array(img_names)
  imagesList, labelsList, namesList = shuffle(images, labels, img_names)
  return imagesList, labelsList, namesList


# Fix shuffling function for these
def makeTrainSet(images, labels, names, validationSize):
  imagesList = images[validationSize:]
  labelsList = labels[validationSize:]
  namesList = names[validationSize:]
  #imagesList, labelsList, namesList = shuffle(imagesList, labelsList, namesList)
  return imagesList, labelsList, namesList

def makeValidationSet(images, labels, names, validationSize):
  imagesList = images[:validationSize]
  labelsList = labels[:validationSize]
  namesList = names[:validationSize]
  #imagesList, labelsList, namesList = shuffle(imagesList, labelsList, namesList)
  return imagesList, labelsList, namesList

def getNewBatch(set, bound, booly):
  xBatch = set[0]
  yBatch = set[1]
  newBound = bound + batchSize
  if booly == 0:
    booly = set[0]
  else:
    booly = set[1]
  if newBound > len(booly):
    newBound = newBound % len(set[0])
    return (xBatch[bound:]), (yBatch[bound:])
  else:
    return xBatch[bound:newBound], yBatch[bound:newBound]

def updateBounds(trainingBound, trainingSet, validBound, validationSet):
  if trainingBound > len(trainingSet[0]):
    trainingBound = (trainingBound + batchSize) % len(trainingSet[0])
  else:
    trainingBound += batchSize
  if validBound > len(validationSet[0]):
    validBound = (validBound + batchSize) % len(validationSet[0])
  else:
    validBound += batchSize
  return trainingBound, validBound

def fixClassLabel(className):
  if "negative" in className:
    className = "negative"
  else:
    className = "positive"
  return className

# Convolutional neural netowrk layer
def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):  
    
  weights = tf.Variable(tf.truncated_normal([conv_filter_size, conv_filter_size, num_input_channels, num_filters],
                         stddev=0.05)) 
  biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

  ## Creating the convolutional layer
  layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
  layer += biases

  ## Max-pooling.  
  layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
  ## Relu: activation function
  layer = tf.nn.relu(layer)

  return layer

def create_flatten_layer(layer):
  # Shape of the layer will be [batch_size img_size img_size num_channels] 
  # Get it from the previous layer.
  layer_shape = layer.get_shape()

  ## Number of features will be img_height * img_width* num_channels.
  num_features = layer_shape[1:4].num_elements()

  ## Flatten the layer so we shall have to reshape to num_features
  layer = tf.reshape(layer, [-1, num_features])

  return layer

# Fully-connected layer
def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
  # Define trainable weights and biases
  weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))

  biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

  # Fully connected layer takes input x and produces wx+b.
  layer = tf.matmul(input, weights) + biases
  if use_relu:
      layer = tf.nn.relu(layer)

  return layer

def show_progress(session, epoch, feed_dict_train, feed_dict_validate, val_loss, accuracy,
                   merged, train_writer, test_writer, i):
  msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
  summary_train, acc = session.run([merged, accuracy], feed_dict=feed_dict_train)
  if math.isnan(acc):
    return
  train_writer.add_summary(summary_train, i)
  summary_test, val_acc = session.run([merged, accuracy], feed_dict=feed_dict_validate)
  if math.isnan(val_acc):
    return
  test_writer.add_summary(summary_test, i)
  print(msg.format(epoch + 1, acc, val_acc, val_loss))

  #val_acc = session.run(accuracy, feed_dict=feed_dict_validate)

def train(session, saver, trainingSet, validationSet, optimiser, cost, accuracy, merged, trainWriter, 
          testWriter, num_iteration, x, y):
  total_iterations = 0
  trainingBound = 0
  validBound = 0
  for i in range(total_iterations,
                  total_iterations + num_iteration):
      # Get next batch in sets and assign them
      x_batch, y_true_batch = getNewBatch(trainingSet, trainingBound, 0)
      x_valid_batch, y_valid_batch = getNewBatch(validationSet, validBound, 1)

      # make sure bounds dont overflow
      trainingBound, validBound = updateBounds(trainingBound, trainingSet, validBound, validationSet)
      
      # feed training and validation batchs
      feed_dict_tr = {x: x_batch, y: y_true_batch}
      feed_dict_val = {x: x_valid_batch, y: y_valid_batch}

      # run training algo
      session.run(optimiser, feed_dict=feed_dict_tr)
      if i % 10 == 0:
          val_loss = session.run(cost, feed_dict=feed_dict_val)
          epoch = int(i/10)
          
          show_progress(session, epoch, feed_dict_tr, feed_dict_val, val_loss,
                        accuracy, merged, trainWriter, testWriter, i)
          saver.save(session, './modelCheckpoints/MURA-model') 
  total_iterations += num_iteration

def main():

  # Path to training set: ("MURA-v1.1\train")
  trainingPath = sys.argv[1]

  # preprocess images by reading from directory using CV2
  images, labels, img_names = preprocessImages(trainingPath)

  # 10% of images for validation
  validationSize = int(0.2*len(images))

  # List holds 
  #   1) List of images
  #   2) List of Labels
  #   3) List of Image names
  # in order for each image processed
  trainingSet = makeTrainSet(images, labels, img_names, validationSize)   #['images','labels','imgnames']
  validationSet = makeValidationSet(images, labels, img_names, validationSize) #['images','labels','imgnames']
  #data = dataset.read_train_sets(train_path, img_size)

  print("Number of files in Training-set:\t\t{}".format(len(trainingSet[1])))
  print("Number of files in Validation-set:\t{}".format(len(validationSet[1])))

  x, y, yClasses = createPlaceholders()
  session = tf.Session()
  layer_conv1 = create_convolutional_layer(input=x,
                num_input_channels=numChannels,
                conv_filter_size=filter_size_conv1,
                num_filters=num_filters_conv1)

  layer_conv2 = create_convolutional_layer(input=layer_conv1,
                num_input_channels=num_filters_conv1,
                conv_filter_size=filter_size_conv2,
                num_filters=num_filters_conv2)

  layer_conv3= create_convolutional_layer(input=layer_conv2,
                num_input_channels=num_filters_conv2,
                conv_filter_size=filter_size_conv3,
                num_filters=num_filters_conv3)
            
  layer_flat = create_flatten_layer(layer_conv3)

  layer_fc1 = create_fc_layer(input=layer_flat,
                      num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                      num_outputs=fc_layer_size,
                      use_relu=True)

  layer_fc2 = create_fc_layer(input=layer_fc1,
                      num_inputs=fc_layer_size,
                      num_outputs=len(categories),
                      use_relu=False)

  y_pred = tf.compat.v1.nn.softmax(layer_fc2,name='y_pred')

  y_pred_cls = tf.compat.v1.argmax(y_pred, dimension=1)
  session.run(tf.global_variables_initializer())
  crossEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y)
  cost = tf.reduce_mean(crossEntropy)
  tf.summary.scalar('cost',cost)
  optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
  correctPrediction = tf.equal(y_pred_cls, yClasses)
  accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
  tf.summary.scalar('accuracy',accuracy)
  merged = tf.compat.v1.summary.merge_all()
  trainWriter = tf.compat.v1.summary.FileWriter('summary/train', session.graph)
  testWriter = tf.compat.v1.summary.FileWriter('summary/test')
  session.run(tf.global_variables_initializer()) 

  saver = tf.compat.v1.train.Saver()

  # 4000 magic number best performance on my pc, but on different devices this number changes... 
  train(session, saver, trainingSet, validationSet, optimiser, cost, accuracy, merged, 
          trainWriter, testWriter , 4000, x, y)

############################
# START OF PROGRAM
main()