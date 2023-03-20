# python program to demonstrate the zooming of the image with the zoom_range argument

# we import all our required libraries
import os
import joblib
import gc

from numpy import expand_dims
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def produce_images(className):
	outputClassDirectory = outputDirectory+ '/' + className
	inputClassDirectory = directory + '/' + className
	if not os.path.exists(outputClassDirectory):
		os.makedirs(outputClassDirectory)

	for inputClassImage in os.scandir(inputClassDirectory):
		baseFileName = inputClassImage.name[0:len(inputClassImage.name) - 4]
		outputFileName = baseFileName + '_aug_' + str(9) + '.jpg'
		if os.path.exists(os.path.join(outputClassDirectory, outputFileName)):
			continue
		gc.collect()
		# we first load the image
		image = load_img(inputClassImage.path)
		# we converting the image which is in PIL format into the numpy array, so that we can apply deep learning methods
		dataImage = img_to_array(image)
		image.close()
		# print(dataImage)
		# expanding dimension of the load image
		imageNew = expand_dims(dataImage, 0)
		# now here below we creating the object of the data augmentation class
		imageDataGen = ImageDataGenerator(zoom_range=0.4, horizontal_flip=True)
		#imageDataGen = ImageDataGenerator(horizontal_flip=True)
		# because as we alreay load image into the memory, so we are using flow() function, to apply transformation
		iterator = imageDataGen.flow(imageNew, batch_size=1)
		# below we generate augmented images and plotting for visualization
		for i in range(10):
			# generating images of each batch
			batch = iterator.next()
			# again we convert back to the unsigned integers value of the image for viewing
			image = array_to_img(batch[0], scale=False)
			outputFileName = baseFileName + '_aug_' + str(i) + '.jpg'
			image.save(os.path.join(outputClassDirectory, outputFileName))
			image.close()
			


directory = 'D:/JukidoStanceImages/jukido_stances'
outputDirectory = directory + '_augmented'

if not os.path.exists(outputDirectory):
	os.makedirs(outputDirectory)

subfolders = [ f.name for f in os.scandir(directory) if f.is_dir() ]

joblib.Parallel(n_jobs=7)(joblib.delayed(produce_images)(className) for className in subfolders)