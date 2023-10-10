#helper for the project jupyter notebook

from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import load_img, img_to_array

#given a filepath to a folder of images 
#returns maxWidth, minWidth, maxHeight, minHeight
def getMaxMin(folder):
    maxWidth = [-1, 0]
    minWidth = [-1, 0]

    maxHeight = [0, -1]
    minHeight = [0, -1]

    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    #iteratively find max/min width/height of images
    for filename in onlyfiles:
        with Image.open(folder + filename) as img:
            width, height = img.size

        #update maxWidth/maxHeight if new max is found
        if width > maxWidth[0]:
            maxWidth = img.size

        if height > maxHeight[1]:
            maxHeight = img.size


        #initialize minWidth/minHeight to first encountered image
        if minWidth[0] < 0:
            minWidth = img.size

        if minHeight[1] < 0:
            minHeight = img.size


        #update minWidth/minHeight if new min is found
        if width < minWidth[0]:
            minWidth = img.size

        if height < minHeight[1]:
            minHeight = img.size

    return maxWidth, minWidth, maxHeight, minHeight

#returns names of all images that match the given dimensions from a folder
def imgSizeMatch(imgSize, folder):

    matchingImgs = []

    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    for filename in onlyfiles:
        with Image.open(folder + filename) as img:
            width, height = img.size

        if width == imgSize[0] and height == imgSize[1]:
            matchingImgs.append(filename)

    return matchingImgs

#iterate through each animal folder and obtain size stats
#returns the max/min Width/Height overall
def calculateExtremeSizes():
    catSizes = getMaxMin("animals/cats/")
    dogSizes = getMaxMin("animals/dogs/")
    pandaSizes = getMaxMin("animals/panda/")

    return [max(catSizes[0], dogSizes[0], pandaSizes[0]), 
                max(catSizes[1], dogSizes[1], pandaSizes[1]), 
                max(catSizes[2], dogSizes[2], pandaSizes[2]), 
                max(catSizes[3], dogSizes[3], pandaSizes[3])]

#returns an array containing the image as an array for all images within the given folder
def getArrays(folder):
    arrayHolder = []
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    for f in onlyfiles:
        img = load_img(folder + f)
        img = img.resize([500,500])
        array = img_to_array(img)
        arrayHolder.append(array.flatten())
    return arrayHolder

#gets avg dimensions of all animal folders
def avgDims():
    allFolders = ["animals/cats/", "animals/dogs/", "animals/panda/"]

    allWidth = 0
    allHeight = 0

    for i in range(0,len(allFolders)):
        folder = allFolders[i]

        onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

        #iteratively calculate dim of all images
        for filename in onlyfiles:
            with Image.open(folder + filename) as img:
                width, height = img.size

            #calculate total
            allWidth += width
            allHeight += height

    return (allWidth / 3000), (allHeight / 3000)