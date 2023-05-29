import cv2 as cv
import numpy as np


def CylinderHighlighter():
    video = cv.VideoCapture("./Vid2.mp4")
    template = cv.imread("./template.png")
    masks = cv.VideoCapture("./Masks.mp4")
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    frameArray = []
    while (video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frameArray.append(frame)
        else:
            break

    masksArray = []
    while (masks.isOpened()):
        ret, frame = masks.read()
        if ret == True:
            masksArray.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        else:
            break


    for num in range(0, len(frameArray)-1):
        frame = frameArray[num]
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = masksArray[num]

        leftMost = 10000
        rightMost = -1
        topMost = 10000
        bottomMost = -1

        for row in range(0, len(mask)-1):
            for col in range(0, len(mask[0])-1):
                if mask[row, col] != 0:
                    if row < topMost:
                        topMost = row
                    if row > bottomMost:
                        bottomMost = row
                    if col < leftMost:
                        leftMost = col
                    if col > rightMost:
                        rightMost = col

        template = cv.bitwise_and(grayFrame, grayFrame, mask=mask)
        croppedTemplate = template[topMost:bottomMost, leftMost:rightMost]
        width, height = croppedTemplate.shape[::-1]

        res = cv.matchTemplate(grayFrame, croppedTemplate, cv.TM_CCOEFF)

        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)

        topLeft = maxLoc
        bottomRight = (topLeft[0] + width, topLeft[1] + height)

        cv.rectangle(frame, topLeft, bottomRight, 255, 2)

        cv.imshow("template", frame)
        cv.waitKey(30)

    videoWriter = cv.VideoWriter('./Output.mp4', 0x7634706d, 28,
                                 (320, 240))
    for frame in frameArray:
        videoWriter.write(frame)
        cv.imshow("", frame)
        cv.waitKey(1)
    videoWriter.release()


def crop_new(arr):  # https://learnopencv.com/cropping-an-image-using-opencv/#cropping-using-opencv

    mask = arr != 0
    n = mask.ndim
    dims = range(n)
    slices = [None]*n

    for i in dims:
        mask_i = mask.any(tuple(dims[:i] + dims[i+1:]))
        slices[i] = (mask_i.argmax(), len(mask_i) - mask_i[::-1].argmax())

    return arr[[slice(*s) for s in slices]]