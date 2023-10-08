#helper for the project jupyter notebook

from PIL import Image
from os import listdir
from os.path import isfile, join

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