import cv2
import os
import shutil
import numpy as np


# Function to extract frames
def FrameCapture(path):
    vidcap = cv2.VideoCapture(path)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    pathFrame = 'C:/Users/Ashik/Desktop/OpenCV Frame Extract Filter/Frames'
    pathFG = 'C:/Users/Ashik/Desktop/OpenCV Frame Extract Filter/Foreground'
    cannyft = 'C:/Users/Ashik/Desktop/OpenCV Frame Extract Filter/CannyFilter'
    success, image = vidcap.read()
    count = 0
    while success:
        # saving original frame
        cv2.imwrite(os.path.join(pathFrame, "frame%d.jpg") % count, image)

        # saving foreground frame
        fgmask = fgbg.apply(image)
        cv2.imwrite(os.path.join(pathFG, "fgframe%d.jpg") % count, fgmask)

        # saving canny filter frame
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite(os.path.join(cannyft, "cannyft%d.jpg") % count, edges)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


# Function to extract frames from webcam
def FrameCaptureFromWebCam():
    vidcap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    pathFrame = 'C:/Users/Ashik/Desktop/OpenCV Frame Extract Filter/Frames'
    pathFG = 'C:/Users/Ashik/Desktop/OpenCV Frame Extract Filter/Foreground'
    cannyft = 'C:/Users/Ashik/Desktop/OpenCV Frame Extract Filter/CannyFilter'
    # Folder cleanup
    shutil.rmtree(pathFrame)
    os.makedirs(pathFrame)
    shutil.rmtree(pathFG)
    os.makedirs(pathFG)
    shutil.rmtree(cannyft)
    os.makedirs(cannyft)
    count = 0
    while True:
        success, image = vidcap.read()
        # saving original frame
        cv2.imwrite(os.path.join(pathFrame, "frame%d.jpg") % count, image)

        # saving foreground frame
        fgmask = fgbg.apply(image)
        cv2.imwrite(os.path.join(pathFG, "fgframe%d.jpg") % count, fgmask)

        # saving canny filter frame
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite(os.path.join(cannyft, "cannyft%d.jpg") % count, edges)
        count += 1
        # concatanate image Horizontally
        img_concate_Hori = np.concatenate((fgmask, edges), axis=1)
        cv2.imshow('Original', image)
        cv2.imshow('Foreground and Canny filter', img_concate_Hori)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # When everything done, release the capture
            vidcap.release()
            cv2.destroyAllWindows()
            break


# Driver Code
if __name__ == '__main__':
    # Calling the function
    # FrameCapture("Missing.mp4")
    FrameCaptureFromWebCam()
