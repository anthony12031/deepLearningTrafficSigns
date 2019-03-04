# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import os, cv2
import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle
import matplotlib.image as mpimg

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    #for c in range(0,43):
    for c in [0,9,28,36]:
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        #gtReader.next() # skip header
        next(gtReader)
        # loop over all images in current annotations file
        counter = 1
        for row in gtReader:
            image = plt.imread(prefix + row[0])
            images.append(image) # the 1th column is the filename
            label = None
            if(c==0):
                label=0
            elif(c==9):
                label=1
            elif(c==28):
                label=2
            elif(c==36):
                label=3
            labels.append(label) # the 8th column is the label
            #guardar imagen para pruebas
            #mpimg.imsave("test_alemania/"+str(label)+"/"+str(counter)+".jpg", image)
            counter+=1
        gtFile.close()
    return images, labels


images,labels=readTrafficSigns("./images")

ALTO = 32
ANCHO = 32
CHANNELS = 3

#reescalar images a 32 x 32
for i in range(0,len(images)):
    images[i]=cv2.resize(images[i], (ALTO, ANCHO), interpolation=cv2.INTER_CUBIC)

images = np.array(images).reshape(-1,ALTO,ANCHO,3)
#serializar imagenes y labels
images_out=open("images.pickle","wb")
pickle.dump(images,images_out)
images_out.close()

labels_out=open("labels.pickle","wb")
pickle.dump(labels,labels_out)
labels_out.close()

