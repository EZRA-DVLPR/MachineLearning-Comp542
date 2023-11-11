#helper for the project jupyter notebook

from PIL import Image
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import load_img, img_to_array

#Input: filepath to a folder containing images
#Output: [maxWidth, minWidth, maxHeight, minHeight] of all images
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

#Input: imgSize - the dimensions of an image [width, height], filepath to a folder containing images
#Output: array of strings containing the names of files
#           eg. [FILENAME1, FILENAME2, ...]
def imgSizeMatch(imgSize, folder):

    matchingImgs = []
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    for filename in onlyfiles:
        with Image.open(folder + filename) as img:
            width, height = img.size

        if width == imgSize[0] and height == imgSize[1]:
            matchingImgs.append(filename)

    return matchingImgs


#Input: filepath to a folder containing images
#Output: [maxWidth, minWidth, maxHeight, MinHeight] of all 3 subfolders (`cats`, `dogs`, and `panda`)
def calculateExtremeSizes(folder):
    catSizes = getMaxMin(folder + "/cats/")
    dogSizes = getMaxMin(folder + "/dogs/")
    pandaSizes = getMaxMin(folder + "/panda/")

    return [max(catSizes[0], dogSizes[0], pandaSizes[0]), 
                min(catSizes[1], dogSizes[1], pandaSizes[1]), 
                max(catSizes[2], dogSizes[2], pandaSizes[2]), 
                min(catSizes[3], dogSizes[3], pandaSizes[3])]

#Input: filepath to a folder containing images
#Output: array containing the images as arrays for all images within the given folder
#           the images given will be reshaped to dimesnsion size (500,500) before being converted into an array
#           eg. [[...IMG1...], [...IMG2...], ...]
def getArrays(folder):
    arrayHolder = []
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    for f in onlyfiles:
        img = load_img(folder + f)
        img = img.resize([500,500])
        array = img_to_array(img)
        arrayHolder.append(array.flatten())
    return arrayHolder

#Input: filepath to a folder containing images
#Output: [avgWidth, avgHeight]
def avgDims(folder):
    allFolders = [folder + "/cats/", folder + "/dogs/", folder + "/panda/"]

    allWidth = 0
    allHeight = 0
    numImages = 0

    #iterate through each subfolder
    for i in range(0, len(allFolders)):
        folder = allFolders[i]
        numImages += len(listdir(allFolders[i]))

        onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

        #iteratively calculate dim of all images
        for filename in onlyfiles:
            with Image.open(folder + filename) as img:
                width, height = img.size

            #calculate total
            allWidth += width
            allHeight += height
    
    #calculate avg
    return (allWidth / numImages), (allHeight / numImages)