import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
import csv

path_image_names_txt = ''
path_images_folder = ''
path_images_save = ''

def ThreshROI(Img, thr):
    height, width = Img.shape
    ImgRes = np.zeros((height, width), dtype=np.uint8)
    ymidI = width/2
    ymidR = width/2
    for i in range(height):
        for j in range(width):
            if j<ymidI or j>ymidR:
                if Img[i,j] > thr:
                    ImgRes[i,j] = 255
                else:
                    ImgRes[i,j] = 0            
        ymidI -= 1
        ymidR += 1
    return(ImgRes)

def RemoveLbl(Img, ImgB, color):
    height, width = Img.shape
    for i in range(height):
        for j in range(width):
            if color == 0:
                if ImgB[i,j] == 0:
                    Img[i,j] = 0
            elif ImgB[i,j] == 255:
                Img[i,j] = 0
    return(Img)

def Preprocessing(img):    
    img = cv.resize(img, (1360,796))
    smooth = cv.GaussianBlur(img, (5,5), 0)
    ret, smoothBin = cv.threshold(smooth, 65, 255, cv.THRESH_BINARY)

    kernel = np.ones((55,55),np.uint8)
    erosion = cv.erode(smoothBin, kernel, iterations = 1)
    dilation = cv.dilate(erosion, kernel, iterations = 1)

    NoLbl = RemoveLbl(img, dilation, 0)
    smoothNoLbl = cv.GaussianBlur(NoLbl, (39,39), 0)
    smoothNoLblThRoi = ThreshROI(smoothNoLbl, 150)
    NoLbl2 = RemoveLbl(NoLbl, smoothNoLblThRoi, 255)
    return(NoLbl2)

def build_filters():
    filters = []
    ksize = 33
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
 
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def rmvEdges(img):
    ret, edgesTh = cv.threshold(img, 240.0, 255.0, cv.THRESH_BINARY)
    kernel = np.ones((20,20),np.uint8)
    dilEdges = cv.dilate(edgesTh, kernel, iterations = 1)
    NoEdges = RemoveLbl(img, dilEdges, 255)
    return(NoEdges)


with open(path_image_names_txt, 'r') as hosts:  
    annotations = csv.reader(hosts)
    for row in annotations:
        imgname = row[0]
        path = path_images_folder + imgname
        print(path)
        img = cv.imread(path, 0)
        
        img = Preprocessing(img)
        filters = build_filters()
        resImg = process(img, filters)
        resImgF = rmvEdges(resImg)

        cv.imwrite(path_images_save + imgname, resImgF)
