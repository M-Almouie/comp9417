import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

categories = ['positive', 'negative']

# Full train_path ("MURA-v1.1\train\XR_BODYPART\patient#\studyN_CLASS\image.png")
def load_train(train_path, image_size):
    images, labels, img_names, cls = []
    print('Processing training images:------------------')
    bodyParts = os.listdir(train_path)
    for bp in bodyParts:
      print("----------------- Reading Body Part: " + bp)
      patients = os.listdir(train_path + "/" + bp)
      for patient in patients:
        classesInDir = os.listdir(train_path + "/" + bp + "/" + patient)
        for fields in classesInDir:
          oldFields = fields 
          fields = fixClassLabel(fields)
          index = categories.index(fields)
          path = os.path.join(train_path, bp, patient, oldFields, '*g')
          files = glob.glob(path)
          for file in files:
              image = cv2.imread(file)
              image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
              image = image.astype(np.float32)
              image = np.multiply(image, 1.0 / 255.0)
              images.append(image)
              label = np.zeros(2)
              label[index] = 1.0
              labels.append(label)
              flbase = os.path.basename(file)
              img_names.append(flbase)
              cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      #assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
  return data_sets

def fixClassLabel(className):
  if "negative" in className:
    className = "negative"
  else:
    className = "positive"
  return className