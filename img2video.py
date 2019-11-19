import cv2
import os
from time import sleep

fps = 30

num = 100

img_size = (3840, 2160)

filelist = sorted(os.listdir('./SDR_540p/'))
im_dir = './data/result/'
video_dir = './data/result_video/'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
for video_name in filelist:

    video_writer = cv2.VideoWriter(video_dir+video_name, fourcc, fps, img_size)
    for i in range(1, num+1):
        im_name = os.path.join(im_dir, video_name.split('.')[0]+'_%03d'%i + '.jpg')
        frame = cv2.imread(im_name)
        video_writer.write(frame)

    video_writer.release()
    print('video %s finish' % video_name)