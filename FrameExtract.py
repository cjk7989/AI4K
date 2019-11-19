import cv2
import os
import shutil

'''
sourcePath = './data/SDR_540p/'
savedPath = './data/low/'
 
filelist = sorted(os.listdir(sourcePath))
for filename in filelist:
    videoCapture = cv2.VideoCapture(sourcePath + filename)
    i = 0
    
    while True:
        success, frame = videoCapture.read()
        if not success:
            print('Video %s is all read, %d frames in total' % (filename, i))
            break
        i += 1
        savedname = "%s_%03d.jpg" % (filename.split('.')[0], i)
        cv2.imwrite(savedPath + savedname, frame)
        
sourcePath = './data/SDR_4K/'
savedPath = './data/high/'
 
filelist = sorted(os.listdir(sourcePath))
for filename in filelist:
    videoCapture = cv2.VideoCapture(sourcePath + filename)
    i = 0
    
    while True:
        success, frame = videoCapture.read()
        if not success:
            print('Video %s is all read, %d frames in total' % (filename, i))
            break
        i += 1
        savedname = "%s_%03d.jpg" % (filename.split('.')[0], i)
        cv2.imwrite(savedPath + savedname, frame)
'''

sourcePath = './SDR_540p/'
savedPath = './data/test/'
 
filelist = sorted(os.listdir(sourcePath))
for filename in filelist:
    videoCapture = cv2.VideoCapture(sourcePath + filename)
    i = 0
    
    while True:
        success, frame = videoCapture.read()
        if not success:
            print('Video %s is all read, %d frames in total' % (filename, i))
            break
        i += 1
        savedname = "%s_%03d.jpg" % (filename.split('.')[0], i)
        cv2.imwrite(savedPath + savedname, frame)