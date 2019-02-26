import numpy as np
import matplotlib.pyplot as plt
from kitti_foundation import Kitti, Kitti_util


velo_path = './velodyne_points/data'

velo = Kitti_util(frame=0, velo_path=velo_path)
frame = velo.velo_file

print(frame.shape)

h_fov, v_fov = (-180, 180), (-24.9, 2.0)
pano_img = velo.velo_2_pano_frame(h_fov, v_fov, depth=False)

# display result image
plt.subplots(1,1, figsize = (13,3) )
plt.title("Result of Vertical FOV ({} , {}) & Horizontal FOV ({} , {})".format(v_fov[0],v_fov[1],h_fov[0],h_fov[1]))
plt.imshow(pano_img)
plt.axis('off')

print(pano_img.shape)


import cv2



velo2 = Kitti_util(frame='all',velo_path=velo_path)
pano = velo2.velo_2_pano(h_fov, v_fov, depth=True)

""" save panorama video """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid = cv2.VideoWriter('pano.avi', fourcc, 25.0, (1030, 66), False)

for frame in pano:
    vid.write(frame)

print('video saved')
vid.release()

# this is result of depth mode = True

from IPython.display import clear_output

""" display video """
vid = cv2.VideoCapture("./pano.avi")

while (True):
    ret, frame = vid.read()
    if not ret:
        vid.release()
        break
    fig = plt.figure(figsize=(12, 3))

    plt.title("Panorama video result")
    plt.axis('off')
    plt.imshow(frame)
    plt.show()
    # clear current frame for next frame
    clear_output(wait=True)

vid.release()

