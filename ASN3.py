import cv2
import numpy
import numpy as np
from numpy import *

def SimpleColorThresholder():
    video = cv2.VideoCapture("./Vid1.mp4")
    frameArray = []
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frameArray.append(frame)
        else:
            break

    videoWriter = cv2.VideoWriter('./SimpleColorThresholder.mp4', 0x7634706d, 28,
                                  (320, 240))

    for frame in frameArray:
        dimensions = frame.shape
        for row in range(0, dimensions[0]):
            for col in range(0, dimensions[1]):
                if notInRange(frame[row, col, 2], frame[row, col, 1], frame[row, col, 0]):
                    frame[row, col, 0] = 0
                    frame[row, col, 1] = 0
                    frame[row, col, 2] = 0
                else:
                    frame[row, col, 0] = 255
                    frame[row, col, 1] = 255
                    frame[row, col, 2] = 255
        cv2.imshow("", frame)
        cv2.waitKey(1)

    for frame in frameArray:
        videoWriter.write(frame)
        cv2.imshow("", frame)
        cv2.waitKey(1)


    videoWriter.release()


def notInRange(red, green, blue):
    rMin = 220
    rMax = 255
    gMin = 165
    gMax = 185
    bMin = 0
    bMax = 50
    if rMin <= red <= rMax:
        return False
    elif gMin <= green <= gMax:
        return False
    elif bMin <= blue <= bMax:
        return False
    return True


def GaussianColorThresholder():
    video = cv2.VideoCapture("./Vid1.mp4")
    frameArray = []
    while (video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frameArray.append(frame)
        else:
            break

    for image in frameArray:
        gaussianFilter(image)
        cv2.imshow("", image)
        cv2.waitKey(25)

    videoWriter = cv2.VideoWriter('./GaussianColorThresholder.mp4', 0x7634706d, 28,
                                  (320, 240))
    for frame in frameArray:
        videoWriter.write(frame)
        cv2.imshow("", frame)
        cv2.waitKey(1)


    videoWriter.release()


def gaussianFilter(image):
    N = image.shape[0]*image.shape[1]
    mu = [0, 0, 0]

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            x = image[row, col]
            mu += x
    mu = (mu/N)[:, None]

    sigma = [0, 0, 0]
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            x = image[row, col][:, None]
            xT = x.T
            xmu = x-mu
            xmuT = xmu.T

            additive = numpy.outer(xmu, xmuT)

            sigma += additive

    sigma = sigma/N

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if pOrange(image, row, col, mu, sigma) > 0.1:
                image[row, col, :] = 0

    return image


def pOrange(frame, row, col, mu, sigma):
    x = frame[row, col]
    x = x[:, None]
    base = 2.718

    xmu = x - mu    
    xmuT = xmu.T
    sigmaInv = numpy.linalg.inv(sigma)
    matmul = numpy.matmul(numpy.matmul(xmuT, sigmaInv), xmu)

    exp = -0.5*matmul

    probabilityOfOrange = base**exp

    return probabilityOfOrange